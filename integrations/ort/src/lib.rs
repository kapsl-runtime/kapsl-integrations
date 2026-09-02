//! Backend-neutral, in-process ONNX Runtime adapter for Kapsl.
//!
//! The exported surface contains only `kapsl-backend-abi` v1 C values. ORT,
//! Rust collections, locks, and tensor ownership stay behind the opaque handle.

use kapsl_backend_abi::*;
use kapsl_core::Manifest;
use kapsl_engine_api::{EngineMetrics, MemoryAllocationClass, MemoryDomain, MemoryReport};
use serde::{Deserialize, Serialize};
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

mod model;
mod preprocess;
mod task;
mod tensor;

use model::{OrtBackend, OrtTuning, SessionPoolStats};
use preprocess::InputPreprocessor;
use task::TaskProcessor;
use tensor::{request_tensors, OwnedTensor};

pub(crate) type FfiError = (i32, String);
pub(crate) type FfiResult<T> = Result<T, FfiError>;

const MAX_BATCH_REQUESTS: usize = 32;
const MAX_SESSION_POOL_SIZE: u32 = 64;
const MAX_SESSION_BUCKETS: usize = 64;
const CAPABILITIES: u64 = KAPSL_BACKEND_CAP_CPU
    | KAPSL_BACKEND_CAP_BATCHING
    | KAPSL_BACKEND_CAP_CANCELLATION
    | KAPSL_BACKEND_CAP_MEMORY_REPORTING
    | KAPSL_BACKEND_CAP_CONCURRENT_INFERENCE;

pub(crate) fn invalid_argument(message: impl Into<String>) -> FfiError {
    (KAPSL_STATUS_INVALID_ARGUMENT, message.into())
}

pub(crate) fn backend_error(message: impl Into<String>) -> FfiError {
    (KAPSL_STATUS_BACKEND_ERROR, message.into())
}

pub(crate) fn cancelled_error(message: impl Into<String>) -> FfiError {
    (KAPSL_STATUS_CANCELLED, message.into())
}

#[derive(Clone, Copy)]
struct HostLogger {
    user_data: usize,
    callback: Option<KapslLogFn>,
}

impl HostLogger {
    fn emit(self, level: u32, message: &str) {
        if let Some(callback) = self.callback {
            // SAFETY: the host callback table and context remain live until
            // adapter shutdown returns; message storage lives for this call.
            unsafe {
                callback(
                    self.user_data as *mut c_void,
                    level,
                    KapslSlice::from_bytes(message.as_bytes()),
                );
            }
        }
    }
}

struct PackState {
    backend: OrtBackend,
    preprocessor: InputPreprocessor,
    task: TaskProcessor,
    logger: HostLogger,
    allocation_id: String,
    metrics: Mutex<MetricsState>,
}

impl PackState {
    fn memory_with_preprocessor(&self, mut report: MemoryReport) -> MemoryReport {
        let bytes = self.preprocessor.resident_bytes();
        if bytes > 0 {
            report.allocations.extend(
                MemoryReport::single(
                    format!("{}:preprocessor", self.allocation_id),
                    MemoryDomain::Host,
                    MemoryAllocationClass::ModelSession,
                    bytes,
                )
                .allocations,
            );
        }
        report
    }

    fn resident_memory_usage(&self) -> usize {
        self.backend
            .loaded_bytes()
            .saturating_add(self.preprocessor.resident_bytes())
    }
}

#[derive(Default)]
struct MetricsState {
    snapshot: EngineMetrics,
    requests: u64,
    failures: u64,
    total_seconds: f64,
}

impl MetricsState {
    fn record(
        &mut self,
        elapsed_seconds: f64,
        request_count: usize,
        success: bool,
        memory_usage: usize,
        pool: SessionPoolStats,
    ) {
        self.requests = self.requests.saturating_add(request_count as u64);
        if !success {
            self.failures = self.failures.saturating_add(request_count as u64);
        }
        self.total_seconds += elapsed_seconds;
        self.snapshot.inference_time = elapsed_seconds;
        self.snapshot.memory_usage = memory_usage;
        self.snapshot.batch_size = request_count;
        self.snapshot.throughput = if self.total_seconds > 0.0 {
            self.requests as f64 / self.total_seconds
        } else {
            0.0
        };
        self.snapshot.error_rate = if self.requests == 0 {
            0.0
        } else {
            self.failures as f64 / self.requests as f64
        };
        self.apply_pool_stats(pool);
        self.snapshot.refresh_timestamp();
    }

    fn snapshot(&mut self, memory_usage: usize, pool: SessionPoolStats) -> EngineMetrics {
        self.snapshot.memory_usage = memory_usage;
        self.apply_pool_stats(pool);
        self.snapshot.refresh_timestamp();
        self.snapshot.clone()
    }

    fn apply_pool_stats(&mut self, pool: SessionPoolStats) {
        self.snapshot.queue_depth = pool.waiting_sessions;
        self.snapshot.onnx_session_pool_total = pool.total_sessions;
        self.snapshot.onnx_session_pool_idle = pool.idle_sessions;
        self.snapshot.onnx_session_pool_waits_total = pool.waits_total;
        self.snapshot.onnx_session_pool_wait_seconds_total = pool.wait_seconds_total;
    }
}

#[derive(Deserialize)]
struct InitOptions {
    provider: String,
    accelerator_profile: String,
    pack_version: String,
    pack_root: PathBuf,
    entrypoint: PathBuf,
    #[serde(default)]
    onnx_tuning: Option<OrtTuning>,
}

#[derive(Serialize)]
struct BatchingPolicy {
    mode: &'static str,
    max_requests: usize,
    self_batches: bool,
    supports_priority: bool,
}

struct ResultOwner {
    name: Vec<u8>,
    tensor: OwnedTensor,
    output: KapslNamedTensorViewV1,
}

struct BatchResultOwner {
    _owners: Vec<ResultOwner>,
    results: Vec<KapslInferenceResultV1>,
}

static API_V1: KapslBackendApiV1 = KapslBackendApiV1 {
    magic: KAPSL_BACKEND_ENTRYPOINT_MAGIC,
    abi_version: KAPSL_BACKEND_ABI_VERSION,
    struct_size: std::mem::size_of::<KapslBackendApiV1>() as u32,
    wire_format: KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1,
    capabilities: CAPABILITIES,
    describe: Some(describe),
    initialize: Some(initialize),
    planned_memory: Some(planned_memory),
    load_model: Some(load_model),
    planned_request_memory: Some(planned_request_memory),
    infer: Some(infer),
    infer_batch: Some(infer_batch),
    infer_stream: None,
    cancel: Some(cancel),
    actual_memory: Some(actual_memory),
    metrics: Some(metrics),
    model_info: Some(model_info),
    kv_capabilities: None,
    kv_topology: None,
    batching_policy: Some(batching_policy),
    health_check: Some(health_check),
    unload: Some(unload),
    shutdown: Some(shutdown),
    release_result: Some(release_result),
    release_batch_result: Some(release_batch_result),
    free_buffer: Some(free_buffer),
};

/// Return the immutable backend-neutral function table.
#[no_mangle]
pub extern "C" fn kapsl_backend_v1() -> *const KapslBackendApiV1 {
    &API_V1
}

unsafe extern "C" fn describe(
    descriptor_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe { clear_buffer(descriptor_out) };
    with_ffi_error(error_out, || {
        let descriptor = serde_json::json!({
            "schema_version": 1,
            "backend": "onnx",
            "adapter_version": env!("CARGO_PKG_VERSION"),
            "backend_abi": KAPSL_BACKEND_ABI_VERSION,
            "wire_format": KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1,
            "execution_mode": "native",
            "profiles": ["cpu"],
            "tasks": ["forward", "embed", "classify", "detect", "transcribe"],
            "preprocessing": ["tensor", "vision", "audio"],
            "runtime": "onnxruntime",
            "runtime_version": "1.23.2",
            "binding": "ort",
            "binding_version": "2.0.0-rc.11",
            "governed_device_memory": false,
            "cancellation": "ort-run-termination",
            "phase": "cpu-inflight-cancellation",
        });
        write_json(descriptor_out, &descriptor)
    })
}

unsafe extern "C" fn initialize(
    config: *const KapslBackendConfigV1,
    handle_out: *mut *mut c_void,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    if !handle_out.is_null() {
        // SAFETY: the caller supplied the handle output slot.
        unsafe { *handle_out = std::ptr::null_mut() };
    }
    with_ffi_error(error_out, || {
        if config.is_null() || handle_out.is_null() {
            return Err(invalid_argument(
                "native ORT initialization requires config and handle outputs",
            ));
        }
        // Read only struct_size until the caller proves the full config is present.
        let struct_size = unsafe { config.cast::<u32>().read() };
        if struct_size < std::mem::size_of::<KapslBackendConfigV1>() as u32 {
            return Err(invalid_argument("native ORT config struct is truncated"));
        }
        // SAFETY: struct_size covers every v1 field read below.
        let config = unsafe { &*config };
        if config.reserved != 0 {
            return Err(invalid_argument(
                "native ORT config has a non-zero reserved field",
            ));
        }
        if config.require_governed_device_memory != 0 {
            return Err(invalid_argument(
                "CPU ORT adapter cannot satisfy governed device memory",
            ));
        }
        let profile = unsafe { required_utf8(config.profile, "profile") }?;
        if profile != "cpu" {
            return Err(invalid_argument(format!(
                "CPU ORT adapter cannot initialize profile `{profile}`"
            )));
        }
        let manifest: Manifest = unsafe { decode_json(config.manifest_json, "model manifest") }?;
        let task = TaskProcessor::from_manifest(&manifest)?;
        let options: InitOptions =
            unsafe { decode_json(config.options_json, "native ORT options") }?;
        validate_options(&options)?;
        validate_tuning(options.onnx_tuning.as_ref())?;
        let logger = unsafe { host_logger(config.host) }?;
        let preprocessor = InputPreprocessor::from_manifest(&manifest)?;
        logger.emit(
            KAPSL_LOG_INFO,
            &format!(
                "initializing ORT {} CPU {} adapter with {} input from {}",
                options.pack_version,
                task.label(),
                preprocessor.label(),
                options.pack_root.display()
            ),
        );
        let state = Box::new(PackState {
            backend: OrtBackend::new(options.onnx_tuning.unwrap_or_default())?,
            preprocessor,
            task,
            logger,
            allocation_id: format!(
                "onnx:{}:{}:host-session",
                config.model_id, config.replica_id
            ),
            metrics: Mutex::new(MetricsState::default()),
        });
        // SAFETY: shutdown consumes this Box exactly once.
        unsafe { *handle_out = Box::into_raw(state).cast() };
        Ok(())
    })
}

unsafe extern "C" fn planned_memory(
    handle: *mut c_void,
    model_path: KapslSlice,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe { clear_buffer(report_out) };
    with_ffi_error(error_out, || {
        let state = unsafe { state(handle) }?;
        let path = unsafe { path_from_slice(model_path) }?;
        let report = state
            .memory_with_preprocessor(state.backend.planned_memory(&path, &state.allocation_id)?);
        write_json(report_out, &report)
    })
}

unsafe extern "C" fn load_model(
    handle: *mut c_void,
    model_path: KapslSlice,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    with_ffi_error(error_out, || {
        let state = unsafe { state(handle) }?;
        let path = unsafe { path_from_slice(model_path) }?;
        state.logger.emit(
            KAPSL_LOG_INFO,
            &format!("loading ONNX model {}", path.display()),
        );
        state.backend.load(&path)?;
        let validation = state
            .backend
            .model_info()
            .and_then(|info| state.preprocessor.validate_model_info(&info));
        if let Err(error) = validation {
            let _ = state.backend.unload();
            return Err(error);
        }
        Ok(())
    })
}

unsafe extern "C" fn planned_request_memory(
    handle: *mut c_void,
    request: *const KapslInferenceRequestV1,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe { clear_buffer(report_out) };
    with_ffi_error(error_out, || {
        let state = unsafe { state(handle) }?;
        let (_, tensors) = unsafe { request_tensors(request) }?;
        let input_bytes = tensors.iter().try_fold(0_usize, |bytes, tensor| {
            bytes
                .checked_add(tensor.data.len())
                .ok_or_else(|| invalid_argument("native ORT request input bytes overflow"))
        })?;
        let bytes = input_bytes
            .checked_add(state.preprocessor.planned_additional_bytes(&tensors)?)
            .ok_or_else(|| invalid_argument("native ORT request memory estimate overflows"))?;
        let report = MemoryReport::single(
            "request:inputs-and-preprocessing",
            MemoryDomain::Host,
            MemoryAllocationClass::RequestTransient,
            bytes,
        );
        write_json(report_out, &report)
    })
}

unsafe extern "C" fn infer(
    handle: *mut c_void,
    request: *const KapslInferenceRequestV1,
    result_out: *mut KapslInferenceResultV1,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    if !result_out.is_null() {
        // SAFETY: the caller supplied a writable result slot.
        unsafe { *result_out = KapslInferenceResultV1::empty() };
    }
    with_ffi_error(error_out, || {
        if result_out.is_null() {
            return Err(invalid_argument("native ORT result output is null"));
        }
        let state = unsafe { state(handle) }?;
        let (request_id, tensors) = unsafe { request_tensors(request) }?;
        let registration = state.backend.register_requests(&[request_id])?;
        if unsafe { request_is_cancelled(request, request_id) } || registration.is_cancelled()? {
            return Err(cancelled_error(
                "native ORT request was cancelled before execution",
            ));
        }
        let started = Instant::now();
        let result = (|| {
            let prepared = state.preprocessor.prepare(&tensors)?;
            if unsafe { request_is_cancelled(request, request_id) }
                || registration.is_cancelled()?
            {
                return Err(cancelled_error(
                    "native ORT request was cancelled during preprocessing",
                ));
            }
            let effective_tensors = prepared
                .as_ref()
                .map_or_else(|| tensors.clone(), |prepared| prepared.views(&tensors));
            state
                .backend
                .infer(&effective_tensors, &registration)
                .and_then(|output| state.task.postprocess(output, &effective_tensors))
        })();
        let elapsed = started.elapsed().as_secs_f64();
        let loaded_bytes = state.resident_memory_usage();
        let pool = state.backend.session_pool_stats();
        state
            .metrics
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .record(elapsed, 1, result.is_ok(), loaded_bytes, pool);
        let tensor = result?;
        if unsafe { request_is_cancelled(request, request_id) } || registration.is_cancelled()? {
            return Err(cancelled_error(
                "native ORT request was cancelled during execution",
            ));
        }
        unsafe { write_result(result_out, tensor) }
    })
}

unsafe extern "C" fn infer_batch(
    handle: *mut c_void,
    batch: *const KapslInferenceBatchV1,
    result_out: *mut KapslInferenceBatchResultV1,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    if !result_out.is_null() {
        // SAFETY: the caller supplied a writable batch-result slot.
        unsafe { *result_out = KapslInferenceBatchResultV1::empty() };
    }
    with_ffi_error(error_out, || {
        if batch.is_null() || result_out.is_null() {
            return Err(invalid_argument(
                "native ORT batch inference requires batch and result outputs",
            ));
        }
        // Read only struct_size before using the complete batch header.
        let struct_size = unsafe { batch.cast::<u32>().read() };
        if struct_size < std::mem::size_of::<KapslInferenceBatchV1>() as u32 {
            return Err(invalid_argument("native ORT batch struct is truncated"));
        }
        // SAFETY: struct_size covers every v1 batch field.
        let batch = unsafe { &*batch };
        let count = usize::try_from(batch.request_count)
            .map_err(|_| invalid_argument("native ORT batch size exceeds this platform"))?;
        if count == 0 || count > MAX_BATCH_REQUESTS || batch.requests.is_null() {
            return Err(invalid_argument(format!(
                "native ORT batch must contain 1..={MAX_BATCH_REQUESTS} requests"
            )));
        }
        let mut request_pointers = Vec::with_capacity(count);
        let mut request_ids = Vec::with_capacity(count);
        let mut tensors = Vec::with_capacity(count);
        for index in 0..count {
            // SAFETY: request_count is bounded and the host promises an array
            // with that many entries for this synchronous call.
            let request = unsafe { batch.requests.add(index) };
            let (request_id, request_tensors) = unsafe { request_tensors(request) }?;
            if unsafe { request_is_cancelled(request, request_id) } {
                return Err((
                    KAPSL_STATUS_CANCELLED,
                    format!("native ORT batch request {request_id} was cancelled before execution"),
                ));
            }
            request_pointers.push(request);
            request_ids.push(request_id);
            tensors.push(request_tensors);
        }

        let state = unsafe { state(handle) }?;
        let registration = state.backend.register_requests(&request_ids)?;
        let started = Instant::now();
        let results = (|| {
            let prepared = tensors
                .iter()
                .map(|inputs| state.preprocessor.prepare(inputs))
                .collect::<FfiResult<Vec<_>>>()?;
            for (request, request_id) in request_pointers
                .iter()
                .copied()
                .zip(request_ids.iter().copied())
            {
                if unsafe { request_is_cancelled(request, request_id) }
                    || registration.is_cancelled()?
                {
                    return Err(cancelled_error(format!(
                        "native ORT batch request {request_id} was cancelled during preprocessing"
                    )));
                }
            }
            let effective_tensors = tensors
                .iter()
                .zip(&prepared)
                .map(|(inputs, prepared)| {
                    prepared
                        .as_ref()
                        .map_or_else(|| inputs.clone(), |prepared| prepared.views(inputs))
                })
                .collect::<Vec<_>>();
            state
                .backend
                .infer_batch(&effective_tensors, &registration)
                .and_then(|outputs| state.task.postprocess_batch(outputs, &effective_tensors))
        })();
        let elapsed = started.elapsed().as_secs_f64();
        let loaded_bytes = state.resident_memory_usage();
        let pool = state.backend.session_pool_stats();
        state
            .metrics
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .record(elapsed, count, results.is_ok(), loaded_bytes, pool);
        let results = results?;
        for (request, request_id) in request_pointers.into_iter().zip(request_ids) {
            if unsafe { request_is_cancelled(request, request_id) }
                || registration.is_cancelled()?
            {
                return Err(cancelled_error(format!(
                    "native ORT batch request {request_id} was cancelled during execution"
                )));
            }
        }
        unsafe { write_batch_result(result_out, results) }
    })
}

unsafe extern "C" fn cancel(handle: *mut c_void, request_id: u64) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        let state = unsafe { state(handle) }?;
        state.backend.cancel(request_id)
    })) {
        Ok(Ok(())) => KAPSL_STATUS_OK,
        Ok(Err((status, message))) => {
            if let Ok(state) = unsafe { state(handle) } {
                state.logger.emit(KAPSL_LOG_ERROR, &message);
            }
            status
        }
        Err(_) => KAPSL_STATUS_PANIC,
    }
}

unsafe extern "C" fn actual_memory(
    handle: *mut c_void,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe { clear_buffer(report_out) };
    with_ffi_error(error_out, || {
        let state = unsafe { state(handle) }?;
        let report =
            state.memory_with_preprocessor(state.backend.actual_memory(&state.allocation_id));
        write_json(report_out, &report)
    })
}

unsafe extern "C" fn metrics(
    handle: *mut c_void,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe { clear_buffer(report_out) };
    with_ffi_error(error_out, || {
        let state = unsafe { state(handle) }?;
        let memory_usage = state.resident_memory_usage();
        let pool = state.backend.session_pool_stats();
        let snapshot = state
            .metrics
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .snapshot(memory_usage, pool);
        write_json(report_out, &snapshot)
    })
}

unsafe extern "C" fn model_info(
    handle: *mut c_void,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe { clear_buffer(report_out) };
    with_ffi_error(error_out, || {
        let state = unsafe { state(handle) }?;
        let mut info = state.backend.model_info()?;
        state.task.adjust_model_info(&mut info);
        write_json(report_out, &info)
    })
}

unsafe extern "C" fn batching_policy(
    handle: *mut c_void,
    report_out: *mut KapslOwnedBuffer,
    error_out: *mut KapslOwnedBuffer,
) -> i32 {
    unsafe { clear_buffer(report_out) };
    with_ffi_error(error_out, || {
        let _state = unsafe { state(handle) }?;
        write_json(
            report_out,
            &BatchingPolicy {
                mode: "request_coalescing",
                max_requests: MAX_BATCH_REQUESTS,
                self_batches: true,
                supports_priority: false,
            },
        )
    })
}

unsafe extern "C" fn health_check(handle: *mut c_void, error_out: *mut KapslOwnedBuffer) -> i32 {
    with_ffi_error(error_out, || {
        let state = unsafe { state(handle) }?;
        if state.backend.is_loaded()? {
            Ok(())
        } else {
            Err(backend_error("ORT model is not loaded"))
        }
    })
}

unsafe extern "C" fn unload(handle: *mut c_void, error_out: *mut KapslOwnedBuffer) -> i32 {
    with_ffi_error(error_out, || {
        let state = unsafe { state(handle) }?;
        state.backend.unload()?;
        state.logger.emit(KAPSL_LOG_INFO, "unloaded ONNX model");
        Ok(())
    })
}

unsafe extern "C" fn shutdown(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: shutdown is the terminal operation and consumes the handle.
        let state = unsafe { Box::from_raw(handle.cast::<PackState>()) };
        state
            .logger
            .emit(KAPSL_LOG_INFO, "shutting down native ORT adapter");
        drop(state);
    }));
}

unsafe extern "C" fn release_result(_handle: *mut c_void, result: *mut KapslInferenceResultV1) {
    if result.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the host returns each successful result exactly once.
        let result = unsafe { &mut *result };
        if !result.owner_context.is_null() {
            // SAFETY: owner_context was created by write_result below.
            unsafe { drop(Box::from_raw(result.owner_context.cast::<ResultOwner>())) };
        }
        *result = KapslInferenceResultV1::empty();
    }));
}

unsafe extern "C" fn release_batch_result(
    _handle: *mut c_void,
    result: *mut KapslInferenceBatchResultV1,
) {
    if result.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the host returns each successful batch result exactly once.
        let result = unsafe { &mut *result };
        if !result.owner_context.is_null() {
            // SAFETY: owner_context was created by write_batch_result below.
            unsafe {
                drop(Box::from_raw(
                    result.owner_context.cast::<BatchResultOwner>(),
                ))
            };
        }
        *result = KapslInferenceBatchResultV1::empty();
    }));
}

unsafe extern "C" fn free_buffer(buffer: KapslOwnedBuffer) {
    if buffer.ptr.is_null() {
        return;
    }
    // SAFETY: buffers originate from into_owned_buffer and are returned once.
    unsafe { drop(Vec::from_raw_parts(buffer.ptr, buffer.len, buffer.capacity)) };
}

fn validate_options(options: &InitOptions) -> FfiResult<()> {
    if !options.provider.eq_ignore_ascii_case("cpu") || options.accelerator_profile != "cpu" {
        return Err(invalid_argument(format!(
            "CPU ORT adapter requires provider/accelerator cpu, received {}/{}",
            options.provider, options.accelerator_profile
        )));
    }
    if options.pack_version.trim().is_empty() {
        return Err(invalid_argument("native ORT pack version may not be empty"));
    }
    let root = options.pack_root.canonicalize().map_err(|error| {
        invalid_argument(format!(
            "resolve native ORT pack root {}: {error}",
            options.pack_root.display()
        ))
    })?;
    let entrypoint = options.entrypoint.canonicalize().map_err(|error| {
        invalid_argument(format!(
            "resolve native ORT entrypoint {}: {error}",
            options.entrypoint.display()
        ))
    })?;
    if !root.is_dir() {
        return Err(invalid_argument("native ORT pack root must be a directory"));
    }
    if !entrypoint.is_file() {
        return Err(invalid_argument(
            "native ORT entrypoint must be a regular file",
        ));
    }
    if !entrypoint.starts_with(root) {
        return Err(invalid_argument(
            "native ORT entrypoint escapes its signed pack root",
        ));
    }
    Ok(())
}

fn validate_tuning(tuning: Option<&OrtTuning>) -> FfiResult<()> {
    let Some(tuning) = tuning else {
        return Ok(());
    };
    if tuning.session_buckets == Some(0)
        || tuning.bucket_dim_granularity == Some(0)
        || tuning.bucket_max_dims == Some(0)
        || tuning.peak_concurrency_hint == Some(0)
    {
        return Err(invalid_argument(
            "native ORT tuning values must be positive when supplied",
        ));
    }
    if tuning
        .session_buckets
        .is_some_and(|value| value > MAX_SESSION_BUCKETS)
    {
        return Err(invalid_argument(format!(
            "native ORT session_buckets may not exceed {MAX_SESSION_BUCKETS}"
        )));
    }
    if tuning
        .peak_concurrency_hint
        .is_some_and(|value| value > MAX_SESSION_POOL_SIZE)
    {
        return Err(invalid_argument(format!(
            "native ORT peak_concurrency_hint may not exceed {MAX_SESSION_POOL_SIZE}"
        )));
    }
    Ok(())
}

unsafe fn host_logger(host: *const KapslBackendHostV1) -> FfiResult<HostLogger> {
    if host.is_null() {
        return Err(invalid_argument("native ORT host table is null"));
    }
    // SAFETY: read only struct_size until the full host table is proven present.
    let struct_size = unsafe { host.cast::<u32>().read() };
    if struct_size < std::mem::size_of::<KapslBackendHostV1>() as u32 {
        return Err(invalid_argument("native ORT host table is truncated"));
    }
    // SAFETY: struct_size covers every v1 host field.
    let host = unsafe { &*host };
    if host.abi_version != KAPSL_BACKEND_ABI_VERSION {
        return Err((
            KAPSL_STATUS_INCOMPATIBLE_ABI,
            format!(
                "native ORT host ABI {} does not match {}",
                host.abi_version, KAPSL_BACKEND_ABI_VERSION
            ),
        ));
    }
    Ok(HostLogger {
        user_data: host.user_data as usize,
        callback: host.log,
    })
}

unsafe fn state<'a>(handle: *mut c_void) -> FfiResult<&'a PackState> {
    if handle.is_null() {
        return Err(invalid_argument("native ORT handle is null"));
    }
    // SAFETY: initialize created this PackState and shutdown has not run.
    Ok(unsafe { &*handle.cast::<PackState>() })
}

unsafe fn required_utf8(slice: KapslSlice, label: &str) -> FfiResult<String> {
    // SAFETY: the caller retains slice storage for this synchronous call.
    let bytes = unsafe { slice.as_bytes() }
        .ok_or_else(|| invalid_argument(format!("native ORT {label} has a null pointer")))?;
    if bytes.is_empty() {
        return Err(invalid_argument(format!(
            "native ORT {label} may not be empty"
        )));
    }
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|error| invalid_argument(format!("native ORT {label} is not UTF-8: {error}")))
}

unsafe fn decode_json<T: serde::de::DeserializeOwned>(
    slice: KapslSlice,
    label: &str,
) -> FfiResult<T> {
    // SAFETY: the caller retains slice storage for this synchronous call.
    let bytes = unsafe { slice.as_bytes() }
        .ok_or_else(|| invalid_argument(format!("native ORT {label} has a null pointer")))?;
    serde_json::from_slice(bytes)
        .map_err(|error| invalid_argument(format!("decode {label} JSON: {error}")))
}

unsafe fn path_from_slice(path: KapslSlice) -> FfiResult<PathBuf> {
    let text = unsafe { required_utf8(path, "model path") }?;
    Ok(Path::new(&text).to_path_buf())
}

unsafe fn request_is_cancelled(request: *const KapslInferenceRequestV1, request_id: u64) -> bool {
    if request.is_null() {
        return true;
    }
    // SAFETY: request validation has succeeded in the current call.
    let request = unsafe { &*request };
    request.is_cancelled.is_some_and(|callback| {
        // SAFETY: the host retains cancellation_context for this call.
        unsafe { callback(request.cancellation_context, request_id) != 0 }
    })
}

unsafe fn write_result(
    result_out: *mut KapslInferenceResultV1,
    tensor: OwnedTensor,
) -> FfiResult<()> {
    if result_out.is_null() {
        return Err(invalid_argument("native ORT result output is null"));
    }
    let owner = Box::new(result_owner(tensor)?);
    let output = &owner.output as *const KapslNamedTensorViewV1;
    let owner_context = Box::into_raw(owner).cast();
    // SAFETY: result_out is a validated writable host slot.
    unsafe {
        *result_out = KapslInferenceResultV1 {
            struct_size: std::mem::size_of::<KapslInferenceResultV1>() as u32,
            output_count: 1,
            outputs: output,
            metadata_json: KapslSlice::empty(),
            owner_context,
        };
    }
    Ok(())
}

unsafe fn write_batch_result(
    result_out: *mut KapslInferenceBatchResultV1,
    tensors: Vec<OwnedTensor>,
) -> FfiResult<()> {
    if result_out.is_null() {
        return Err(invalid_argument("native ORT batch result output is null"));
    }
    let owners = tensors
        .into_iter()
        .map(result_owner)
        .collect::<FfiResult<Vec<_>>>()?;
    let results = owners
        .iter()
        .map(|owner| KapslInferenceResultV1 {
            struct_size: std::mem::size_of::<KapslInferenceResultV1>() as u32,
            output_count: 1,
            outputs: &owner.output,
            metadata_json: KapslSlice::empty(),
            owner_context: std::ptr::null_mut(),
        })
        .collect::<Vec<_>>();
    let mut owner = Box::new(BatchResultOwner {
        _owners: owners,
        results,
    });
    let result_count = u32::try_from(owner.results.len())
        .map_err(|_| backend_error("ORT batch result count exceeds backend ABI v1"))?;
    let results = owner.results.as_ptr();
    let owner_context = (&mut *owner as *mut BatchResultOwner).cast();
    std::mem::forget(owner);
    // SAFETY: result_out is a validated writable host slot.
    unsafe {
        *result_out = KapslInferenceBatchResultV1 {
            struct_size: std::mem::size_of::<KapslInferenceBatchResultV1>() as u32,
            result_count,
            results,
            owner_context,
        };
    }
    Ok(())
}

fn result_owner(tensor: OwnedTensor) -> FfiResult<ResultOwner> {
    let mut owner = ResultOwner {
        name: tensor.name.as_bytes().to_vec(),
        tensor,
        output: KapslNamedTensorViewV1 {
            struct_size: std::mem::size_of::<KapslNamedTensorViewV1>() as u32,
            reserved: 0,
            name: KapslSlice::empty(),
            tensor: KapslTensorViewV1::empty(),
        },
    };
    owner.output.name = KapslSlice::from_bytes(&owner.name);
    owner.output.tensor = KapslTensorViewV1 {
        struct_size: std::mem::size_of::<KapslTensorViewV1>() as u32,
        dtype: owner.tensor.dtype,
        memory_kind: KAPSL_MEMORY_HOST,
        flags: KAPSL_TENSOR_FLAG_CONTIGUOUS | KAPSL_TENSOR_FLAG_READ_ONLY,
        device_id: -1,
        rank: u32::try_from(owner.tensor.shape.len())
            .map_err(|_| backend_error("ORT output rank exceeds backend ABI v1"))?,
        shape: owner.tensor.shape.as_ptr(),
        strides: std::ptr::null(),
        data: owner.tensor.data.as_ptr().cast(),
        byte_len: u64::try_from(owner.tensor.data.len())
            .map_err(|_| backend_error("ORT output byte length exceeds backend ABI v1"))?,
    };
    Ok(owner)
}

unsafe fn clear_buffer(output: *mut KapslOwnedBuffer) {
    if !output.is_null() {
        // SAFETY: the caller supplied a writable output slot.
        unsafe { *output = KapslOwnedBuffer::empty() };
    }
}

fn into_owned_buffer(mut bytes: Vec<u8>) -> KapslOwnedBuffer {
    if bytes.is_empty() {
        return KapslOwnedBuffer::empty();
    }
    let output = KapslOwnedBuffer {
        ptr: bytes.as_mut_ptr(),
        len: bytes.len(),
        capacity: bytes.capacity(),
    };
    std::mem::forget(bytes);
    output
}

fn write_json<T: Serialize>(output: *mut KapslOwnedBuffer, value: &T) -> FfiResult<()> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| backend_error(format!("encode native ORT JSON: {error}")))?;
    write_buffer(output, bytes)
}

fn write_buffer(output: *mut KapslOwnedBuffer, bytes: Vec<u8>) -> FfiResult<()> {
    if output.is_null() {
        return Err(invalid_argument("native ORT output buffer is null"));
    }
    // SAFETY: output was checked and the caller owns this output slot.
    unsafe { *output = into_owned_buffer(bytes) };
    Ok(())
}

fn with_ffi_error(
    error_out: *mut KapslOwnedBuffer,
    operation: impl FnOnce() -> FfiResult<()>,
) -> i32 {
    unsafe { clear_buffer(error_out) };
    let result = catch_unwind(AssertUnwindSafe(operation));
    let error = match result {
        Ok(Ok(())) => return KAPSL_STATUS_OK,
        Ok(Err(error)) => error,
        Err(_) => (
            KAPSL_STATUS_PANIC,
            "native ORT adapter caught a panic".to_string(),
        ),
    };
    if !error_out.is_null() {
        // SAFETY: the caller supplied a writable error slot.
        unsafe { *error_out = into_owned_buffer(error.1.into_bytes()) };
    }
    error.0
}

#[cfg(test)]
mod tests;
