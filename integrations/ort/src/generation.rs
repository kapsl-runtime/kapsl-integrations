//! ONNX autoregressive generation behind backend ABI v1.
//!
//! The stable C boundary remains tensor-only. This module translates that
//! wire contract into the published Kapsl LLM engine and translates its async
//! UTF-8 delta stream back into borrowed ABI chunks. Accelerator profiles bind
//! the published LLM allocation-scope provider to this adapter's scoped ABI
//! allocator so model, replica, request, and request-batch ownership remain
//! explicit without introducing another process or tensor serialization.

use crate::tensor::{request_tensors, OwnedTensor};
#[cfg(any(feature = "profile-cuda12", feature = "profile-tensorrt10"))]
use crate::{
    allocator::{AllocationScopeBridge, AllocatorLease, ClientKey, HostDeviceCallbacks},
    profile::COMPILED_PROFILE,
};
use crate::{backend_error, cancelled_error, invalid_argument, result_owner, FfiResult};
use futures::StreamExt;
use kapsl_backend_abi::{
    KapslBackendStreamChunkFn, KapslInferenceRequestV1, KapslInferenceResultV1, KAPSL_DTYPE_UTF8,
    KAPSL_STATUS_BACKEND_ERROR, KAPSL_STATUS_CANCELLED, KAPSL_STATUS_INVALID_ARGUMENT,
    KAPSL_STATUS_OK, KAPSL_STATUS_PANIC,
};
use kapsl_engine_api::{
    BinaryTensorPacket, CancellationToken, Engine, EngineError, EngineMetrics, EngineModelInfo,
    EngineStream, InferenceRequest, MemoryReport, RequestMetadata, TensorDtype,
};
use kapsl_llm::llm_backend::LLMBackend;
use serde::Deserialize;
use std::collections::HashMap;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
#[cfg(any(feature = "profile-cuda12", feature = "profile-tensorrt10"))]
use std::sync::Arc;
use std::sync::{mpsc, Mutex, MutexGuard, OnceLock};

const MAX_METADATA_JSON_BYTES: usize = 8 * 1024 * 1024;
const MAX_SESSION_ID_BYTES: usize = 4 * 1024;
const MAX_STOP_TOKEN_IDS: usize = 4 * 1024;

#[derive(Default, Deserialize)]
struct RequestEnvelope {
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    metadata: Option<RequestMetadata>,
}

struct RequestRegistration<'a> {
    active: &'a Mutex<HashMap<u64, CancellationToken>>,
    request_id: u64,
    cancellation: CancellationToken,
}

impl RequestRegistration<'_> {
    fn cancellation(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    fn cancel(&self) {
        self.cancellation.cancel();
    }

    fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
}

impl Drop for RequestRegistration<'_> {
    fn drop(&mut self) {
        self.active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&self.request_id);
    }
}

pub(crate) struct GenerationBackend {
    engine: Mutex<Option<LLMBackend>>,
    active_requests: Mutex<HashMap<u64, CancellationToken>>,
    loaded_model: Mutex<Option<PathBuf>>,
    #[cfg(any(feature = "profile-cuda12", feature = "profile-tensorrt10"))]
    _allocator_lease: AllocatorLease,
}

impl GenerationBackend {
    #[cfg(feature = "profile-cpu")]
    pub(crate) fn new_cpu() -> FfiResult<Self> {
        generation_load_runtime()?;
        Ok(Self {
            engine: Mutex::new(Some(LLMBackend::with_device("cpu".to_string(), 0))),
            active_requests: Mutex::new(HashMap::new()),
            loaded_model: Mutex::new(None),
        })
    }

    #[cfg(any(feature = "profile-cuda12", feature = "profile-tensorrt10"))]
    pub(crate) fn new_accelerator(
        device_id: u32,
        model_id: u32,
        replica_id: u32,
        callbacks: HostDeviceCallbacks,
    ) -> FfiResult<Self> {
        generation_load_runtime()?;
        let device_id = i32::try_from(device_id)
            .map_err(|_| invalid_argument("ORT CUDA device ID exceeds i32"))?;
        let client = ClientKey::new(model_id, replica_id);
        let allocator_lease = crate::allocator::register_client(device_id, client, callbacks)
            .map_err(backend_error)?;
        let scope_provider = Arc::new(AllocationScopeBridge::new(device_id, client));
        let engine = LLMBackend::with_device(COMPILED_PROFILE.provider().to_string(), device_id)
            .with_device_allocation_scope_provider(model_id, replica_id, scope_provider);
        Ok(Self {
            engine: Mutex::new(Some(engine)),
            active_requests: Mutex::new(HashMap::new()),
            loaded_model: Mutex::new(None),
            _allocator_lease: allocator_lease,
        })
    }

    pub(crate) fn planned_memory(&self, model_path: &Path) -> FfiResult<MemoryReport> {
        self.with_engine(|engine| engine.planned_memory(model_path))
    }

    pub(crate) fn load(&self, model_path: &Path) -> FfiResult<()> {
        let canonical = model_path.canonicalize().map_err(|error| {
            backend_error(format!(
                "resolve ONNX generation model {}: {error}",
                model_path.display()
            ))
        })?;
        if self
            .loaded_model
            .lock()
            .map_err(|_| backend_error("ORT generation model-path lock is poisoned"))?
            .is_some()
        {
            return Err(backend_error(
                "ORT generation model is already loaded; unload it before loading another model",
            ));
        }

        let mut engine = self
            .engine
            .lock()
            .map_err(|_| backend_error("ORT generation engine lock is poisoned"))?
            .take()
            .ok_or_else(|| backend_error("ORT generation engine is busy"))?;
        let load_path = canonical.clone();
        let (sender, receiver) = mpsc::sync_channel(1);
        generation_load_runtime()?.spawn(async move {
            let result = engine.load(&load_path).await;
            if result.is_err() {
                engine.unload();
            }
            let _ = sender.send((engine, result));
        });
        let (engine, result) = receiver.recv().map_err(|_| {
            backend_error("ORT generation load task ended without returning its engine")
        })?;
        *self
            .engine
            .lock()
            .map_err(|_| backend_error("ORT generation engine lock is poisoned"))? = Some(engine);
        result.map_err(engine_error)?;
        *self
            .loaded_model
            .lock()
            .map_err(|_| backend_error("ORT generation model-path lock is poisoned"))? =
            Some(canonical);
        Ok(())
    }

    pub(crate) unsafe fn planned_request_memory(
        &self,
        request: *const KapslInferenceRequestV1,
    ) -> FfiResult<MemoryReport> {
        let request = unsafe { decode_request(request, CancellationToken::new()) }?;
        self.with_engine(|engine| Ok(engine.planned_request_memory(&request)))
    }

    pub(crate) unsafe fn infer(
        &self,
        request: *const KapslInferenceRequestV1,
    ) -> FfiResult<OwnedTensor> {
        let (request_id, registration) = unsafe { self.register_request(request) }?;
        let inference_request = unsafe { decode_request(request, registration.cancellation()) }?;
        if unsafe { crate::request_is_cancelled(request, request_id) } {
            registration.cancel();
        }
        if registration.is_cancelled() {
            return Err(cancelled_error(format!(
                "native ORT generation request {request_id} was cancelled before execution"
            )));
        }

        let mut stream = self.engine_stream(&inference_request, request_id)?;
        let result = futures::executor::block_on(async {
            let mut output = Vec::new();
            let mut saw_chunk = false;
            while let Some(chunk) = stream.next().await {
                if unsafe { crate::request_is_cancelled(request, request_id) } {
                    registration.cancel();
                }
                let chunk = chunk.map_err(engine_error)?;
                let tensor = packet_tensor(chunk)?;
                saw_chunk = true;
                output.extend_from_slice(&tensor.data);
            }
            if registration.is_cancelled() {
                return Err(cancelled_error(format!(
                    "native ORT generation request {request_id} was cancelled during execution"
                )));
            }
            if !saw_chunk {
                return Err(backend_error("ORT generation produced no output chunks"));
            }
            Ok(OwnedTensor {
                name: "token".to_string(),
                dtype: KAPSL_DTYPE_UTF8,
                shape: vec![1, output.len() as i64],
                data: output,
            })
        });
        result
    }

    pub(crate) unsafe fn infer_stream(
        &self,
        request: *const KapslInferenceRequestV1,
        user_data: *mut c_void,
        on_chunk: KapslBackendStreamChunkFn,
    ) -> FfiResult<()> {
        let (request_id, registration) = unsafe { self.register_request(request) }?;
        let inference_request = unsafe { decode_request(request, registration.cancellation()) }?;
        if unsafe { crate::request_is_cancelled(request, request_id) } {
            registration.cancel();
        }
        if registration.is_cancelled() {
            return Err(cancelled_error(format!(
                "native ORT generation request {request_id} was cancelled before execution"
            )));
        }

        let mut stream = self.engine_stream(&inference_request, request_id)?;
        futures::executor::block_on(async {
            while let Some(chunk) = stream.next().await {
                if unsafe { crate::request_is_cancelled(request, request_id) } {
                    registration.cancel();
                }
                if registration.is_cancelled() {
                    await_cancellation_acknowledgement(&mut stream).await;
                    return Err(cancelled_error(format!(
                        "native ORT generation request {request_id} was cancelled during execution"
                    )));
                }
                let owner = result_owner(packet_tensor(chunk.map_err(engine_error)?)?)?;
                let result = KapslInferenceResultV1 {
                    struct_size: std::mem::size_of::<KapslInferenceResultV1>() as u32,
                    output_count: 1,
                    outputs: &owner.output,
                    metadata_json: kapsl_backend_abi::KapslSlice::empty(),
                    owner_context: std::ptr::null_mut(),
                };
                let callback_status = catch_unwind(AssertUnwindSafe(|| {
                    // SAFETY: the owner retains all borrowed result storage for
                    // this callback invocation.
                    unsafe { on_chunk(user_data, request_id, &result) }
                }))
                .unwrap_or(KAPSL_STATUS_PANIC);
                if callback_status != KAPSL_STATUS_OK {
                    registration.cancel();
                    await_cancellation_acknowledgement(&mut stream).await;
                    return Err(callback_error(callback_status));
                }
            }
            if registration.is_cancelled() {
                return Err(cancelled_error(format!(
                    "native ORT generation request {request_id} was cancelled during execution"
                )));
            }
            Ok(())
        })
    }

    pub(crate) fn cancel(&self, request_id: u64) -> FfiResult<()> {
        let requests = self
            .active_requests
            .lock()
            .map_err(|_| backend_error("ORT generation cancellation registry is poisoned"))?;
        if let Some(cancellation) = requests.get(&request_id) {
            cancellation.cancel();
        }
        Ok(())
    }

    pub(crate) fn actual_memory(&self) -> FfiResult<MemoryReport> {
        self.with_engine(|engine| Ok(engine.actual_memory()))
    }

    pub(crate) fn metrics(&self) -> FfiResult<EngineMetrics> {
        self.with_engine(|engine| Ok(engine.metrics()))
    }

    pub(crate) fn model_info(&self) -> FfiResult<EngineModelInfo> {
        self.with_engine(|engine| engine.health_check())?;
        Ok(EngineModelInfo {
            input_names: vec!["input".to_string()],
            output_names: vec!["token".to_string()],
            input_shapes: vec![vec![-1, -1]],
            output_shapes: vec![vec![-1, -1]],
            input_dtypes: vec!["string".to_string()],
            output_dtypes: vec!["string".to_string()],
            framework: Some("onnx".to_string()),
            model_version: None,
            peak_concurrency: Some(1),
        })
    }

    pub(crate) fn is_loaded(&self) -> FfiResult<bool> {
        self.with_engine(|engine| match engine.health_check() {
            Ok(()) => Ok(true),
            Err(EngineError::ModelNotLoaded) => Ok(false),
            Err(error) => Err(error),
        })
    }

    pub(crate) fn unload(&self) -> FfiResult<()> {
        for cancellation in self
            .active_requests
            .lock()
            .map_err(|_| backend_error("ORT generation cancellation registry is poisoned"))?
            .values()
        {
            cancellation.cancel();
        }
        let mut slot = self
            .engine
            .lock()
            .map_err(|_| backend_error("ORT generation engine lock is poisoned"))?;
        let engine = slot
            .as_mut()
            .ok_or_else(|| backend_error("ORT generation engine is busy"))?;
        engine.unload();
        *self
            .loaded_model
            .lock()
            .map_err(|_| backend_error("ORT generation model-path lock is poisoned"))? = None;
        Ok(())
    }

    unsafe fn register_request(
        &self,
        request: *const KapslInferenceRequestV1,
    ) -> FfiResult<(u64, RequestRegistration<'_>)> {
        let (request_id, _) = unsafe { request_tensors(request) }?;
        if request_id == 0 {
            return Err(invalid_argument(
                "native ORT generation requires a non-zero request ID",
            ));
        }
        let cancellation = CancellationToken::new();
        let mut active = self
            .active_requests
            .lock()
            .map_err(|_| backend_error("ORT generation cancellation registry is poisoned"))?;
        match active.entry(request_id) {
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(invalid_argument(format!(
                    "native ORT generation request ID {request_id} is already active"
                )))
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(cancellation.clone());
            }
        }
        drop(active);
        Ok((
            request_id,
            RequestRegistration {
                active: &self.active_requests,
                request_id,
                cancellation,
            },
        ))
    }

    fn engine_stream(
        &self,
        request: &InferenceRequest,
        request_id: u64,
    ) -> FfiResult<EngineStream> {
        let engine = self.engine_lock()?;
        let engine = engine
            .as_ref()
            .ok_or_else(|| backend_error("ORT generation engine is busy"))?;
        Ok(engine.infer_stream_with_allocation_request_id(request, request_id))
    }

    fn with_engine<T>(
        &self,
        operation: impl FnOnce(&LLMBackend) -> Result<T, EngineError>,
    ) -> FfiResult<T> {
        let engine = self.engine_lock()?;
        let engine = engine
            .as_ref()
            .ok_or_else(|| backend_error("ORT generation engine is busy"))?;
        operation(engine).map_err(engine_error)
    }

    fn engine_lock(&self) -> FfiResult<MutexGuard<'_, Option<LLMBackend>>> {
        self.engine
            .lock()
            .map_err(|_| backend_error("ORT generation engine lock is poisoned"))
    }
}

async fn await_cancellation_acknowledgement(stream: &mut EngineStream) {
    // kapsl-llm keeps a cancelled stream alive until its scheduler has emitted
    // terminal output, which makes the session safe to reuse synchronously.
    while stream.next().await.is_some() {}
}

fn generation_load_runtime() -> FfiResult<&'static tokio::runtime::Runtime> {
    static RUNTIME: OnceLock<Result<tokio::runtime::Runtime, String>> = OnceLock::new();
    match RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("kapsl-ort-generation-load")
            .enable_all()
            .build()
            .map_err(|error| format!("create ORT generation runtime: {error}"))
    }) {
        Ok(runtime) => Ok(runtime),
        Err(message) => Err(backend_error(message.clone())),
    }
}

impl Drop for GenerationBackend {
    fn drop(&mut self) {
        for cancellation in self
            .active_requests
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values()
        {
            cancellation.cancel();
        }
        if let Some(engine) = self
            .engine
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_mut()
        {
            engine.unload();
        }
    }
}

unsafe fn decode_request(
    request: *const KapslInferenceRequestV1,
    cancellation: CancellationToken,
) -> FfiResult<InferenceRequest> {
    let (request_id, tensors) = unsafe { request_tensors(request) }?;
    if tensors.len() != 1 || tensors[0].name != "input" {
        return Err(invalid_argument(
            "native ORT generation requires exactly one UTF-8 tensor named `input`",
        ));
    }
    let input = tensors[0];
    if input.dtype != KAPSL_DTYPE_UTF8 {
        return Err(invalid_argument(
            "native ORT generation requires a UTF-8 input tensor",
        ));
    }
    std::str::from_utf8(input.data).map_err(|error| {
        invalid_argument(format!(
            "native ORT generation input is not valid UTF-8: {error}"
        ))
    })?;

    // SAFETY: request_tensors validated the complete v1 request header.
    let request_ref = unsafe { &*request };
    if request_ref.metadata_json.len > MAX_METADATA_JSON_BYTES {
        return Err(invalid_argument(format!(
            "native ORT generation metadata exceeds {MAX_METADATA_JSON_BYTES} bytes"
        )));
    }
    // SAFETY: the host retains metadata storage for this synchronous call.
    let metadata_bytes = unsafe { request_ref.metadata_json.as_bytes() }
        .ok_or_else(|| invalid_argument("native ORT generation metadata has a null pointer"))?;
    let mut envelope = if metadata_bytes.is_empty() {
        RequestEnvelope::default()
    } else {
        serde_json::from_slice::<RequestEnvelope>(metadata_bytes).map_err(|error| {
            invalid_argument(format!(
                "decode native ORT generation request metadata: {error}"
            ))
        })?
    };
    validate_envelope(&envelope)?;
    let mut metadata = envelope.metadata.take().unwrap_or_default();
    if metadata.request_id.is_none() {
        metadata.request_id = Some(format!("abi-{request_id}"));
    }

    Ok(InferenceRequest {
        input: BinaryTensorPacket {
            shape: input.shape.to_vec(),
            dtype: TensorDtype::Utf8,
            data: input.data.to_vec(),
        },
        additional_inputs: Vec::new(),
        session_id: envelope.session_id,
        metadata: Some(metadata),
        cancellation: Some(cancellation),
    })
}

fn validate_envelope(envelope: &RequestEnvelope) -> FfiResult<()> {
    if let Some(session_id) = envelope.session_id.as_deref() {
        if session_id.trim().is_empty() || session_id.len() > MAX_SESSION_ID_BYTES {
            return Err(invalid_argument(format!(
                "native ORT generation session ID must contain 1..={MAX_SESSION_ID_BYTES} bytes"
            )));
        }
    }
    let Some(metadata) = envelope.metadata.as_ref() else {
        return Ok(());
    };
    if metadata
        .stop_token_ids
        .as_ref()
        .is_some_and(|ids| ids.len() > MAX_STOP_TOKEN_IDS)
    {
        return Err(invalid_argument(format!(
            "native ORT generation stop-token list may contain at most {MAX_STOP_TOKEN_IDS} entries"
        )));
    }
    if metadata
        .temperature
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        return Err(invalid_argument(
            "native ORT generation temperature must be finite and non-negative",
        ));
    }
    if metadata
        .top_p
        .is_some_and(|value| !value.is_finite() || value <= 0.0 || value > 1.0)
    {
        return Err(invalid_argument(
            "native ORT generation top_p must be finite and in (0, 1]",
        ));
    }
    if metadata
        .repetition_penalty
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err(invalid_argument(
            "native ORT generation repetition_penalty must be finite and positive",
        ));
    }
    if let (Some(minimum), Some(maximum)) = (metadata.min_new_tokens, metadata.max_new_tokens) {
        if maximum > 0 && minimum > maximum {
            return Err(invalid_argument(
                "native ORT generation min_new_tokens may not exceed max_new_tokens",
            ));
        }
    }
    Ok(())
}

fn packet_tensor(packet: BinaryTensorPacket) -> FfiResult<OwnedTensor> {
    packet.validate().map_err(engine_error)?;
    if packet.dtype != TensorDtype::Utf8 {
        return Err(backend_error(format!(
            "ORT generation returned {}, expected string",
            packet.dtype
        )));
    }
    std::str::from_utf8(&packet.data).map_err(|error| {
        backend_error(format!("ORT generation returned invalid UTF-8: {error}"))
    })?;
    Ok(OwnedTensor {
        name: "token".to_string(),
        dtype: KAPSL_DTYPE_UTF8,
        shape: packet.shape,
        data: packet.data,
    })
}

fn engine_error(error: EngineError) -> crate::FfiError {
    let status = match error {
        EngineError::InvalidInput { .. } => KAPSL_STATUS_INVALID_ARGUMENT,
        EngineError::Cancelled { .. } => KAPSL_STATUS_CANCELLED,
        _ => KAPSL_STATUS_BACKEND_ERROR,
    };
    (status, error.to_string())
}

fn callback_error(status: i32) -> crate::FfiError {
    let mapped = match status {
        KAPSL_STATUS_CANCELLED => KAPSL_STATUS_CANCELLED,
        KAPSL_STATUS_INVALID_ARGUMENT => KAPSL_STATUS_INVALID_ARGUMENT,
        KAPSL_STATUS_PANIC => KAPSL_STATUS_PANIC,
        _ => KAPSL_STATUS_BACKEND_ERROR,
    };
    (
        mapped,
        format!("native ORT stream consumer returned status {status}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_metadata_validation_is_bounded() {
        let valid = RequestEnvelope {
            session_id: Some("session".to_string()),
            metadata: Some(RequestMetadata {
                max_new_tokens: Some(4),
                min_new_tokens: Some(2),
                temperature: Some(0.0),
                top_p: Some(1.0),
                repetition_penalty: Some(1.0),
                ..RequestMetadata::default()
            }),
        };
        assert!(validate_envelope(&valid).is_ok());

        let invalid = RequestEnvelope {
            session_id: valid.session_id,
            metadata: Some(RequestMetadata {
                max_new_tokens: Some(1),
                min_new_tokens: Some(2),
                ..RequestMetadata::default()
            }),
        };
        assert!(validate_envelope(&invalid).is_err());
    }
}
