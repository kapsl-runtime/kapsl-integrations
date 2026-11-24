//! Real-GPU checks for the CUDA 12 profile.
//!
//! The host suite drives the governed allocator with host memory standing in
//! for device memory. These tests back the same ABI host callbacks with
//! `cudaMalloc`/`cudaFree`, so ORT executes on the GPU through the engine's
//! allocation contract. They need an NVIDIA GPU, `libcudart.so.12`, and the
//! governed runtime in `ORT_LIB_LOCATION`, like the host suite:
//!
//! ```sh
//! ORT_LIB_LOCATION=/path/to/governed/cuda12 cargo test -p kapsl-backend-ort \
//!     --no-default-features --features profile-cuda12 gpu_tests -- --ignored --test-threads=1
//! ```

use super::*;
use std::collections::HashMap;
use std::ptr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

type CudaMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> i32;
type CudaFree = unsafe extern "C" fn(*mut c_void) -> i32;
type CudaSynchronize = unsafe extern "C" fn() -> i32;

struct Cuda {
    malloc: CudaMalloc,
    free: CudaFree,
    synchronize: CudaSynchronize,
    _library: libloading::Library,
}

impl Cuda {
    fn load() -> Self {
        // SAFETY: loading the CUDA runtime only runs its library initialisers.
        let library =
            unsafe { libloading::Library::new("libcudart.so.12") }.expect("load libcudart.so.12");
        // SAFETY: these are the documented CUDA runtime signatures, and the
        // library outlives the copied function pointers.
        unsafe {
            let malloc = *library.get::<CudaMalloc>(b"cudaMalloc\0").unwrap();
            let free = *library.get::<CudaFree>(b"cudaFree\0").unwrap();
            let synchronize = *library
                .get::<CudaSynchronize>(b"cudaDeviceSynchronize\0")
                .unwrap();
            Self {
                malloc,
                free,
                synchronize,
                _library: library,
            }
        }
    }
}

struct LiveAllocation {
    pointer: usize,
    bytes: u64,
}

#[derive(Default)]
struct Ledger {
    live: HashMap<u64, LiveAllocation>,
    weights: usize,
    workspace: usize,
    other: usize,
}

/// Engine stand-in whose governed allocations are real device memory.
struct GpuHost {
    cuda: Cuda,
    next_id: AtomicU64,
    unscoped_allocations: AtomicUsize,
    rejected_frees: AtomicUsize,
    ledger: Mutex<Ledger>,
}

impl GpuHost {
    fn ledger(&self) -> std::sync::MutexGuard<'_, Ledger> {
        self.ledger
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

unsafe extern "C" fn gpu_allocate_scoped(
    user_data: *mut c_void,
    request: *const KapslScopedDeviceAllocationRequestV1,
    allocation_out: *mut KapslDeviceAllocationV1,
) -> i32 {
    if user_data.is_null() || request.is_null() || allocation_out.is_null() {
        return KAPSL_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the fixture keeps its GpuHost alive until shutdown returns.
    let host = unsafe { &*user_data.cast::<GpuHost>() };
    // SAFETY: the adapter passes a readable request for this call.
    let request = unsafe { *request };
    // cudaMalloc returns at least 256-byte alignment.
    if !request.is_well_formed() || request.alignment > 256 {
        return KAPSL_STATUS_INVALID_ARGUMENT;
    }
    let Ok(bytes) = usize::try_from(request.bytes) else {
        return KAPSL_STATUS_BACKEND_ERROR;
    };
    let mut pointer = ptr::null_mut();
    // SAFETY: pointer is a writable out-slot for the CUDA runtime.
    if unsafe { (host.cuda.malloc)(&mut pointer, bytes) } != 0 || (pointer.is_null() && bytes != 0)
    {
        return KAPSL_STATUS_BACKEND_ERROR;
    }
    let allocation_id = host.next_id.fetch_add(1, Ordering::Relaxed) + 1;
    {
        let mut ledger = host.ledger();
        match request.allocation_class {
            KAPSL_ALLOCATION_CLASS_WEIGHTS => ledger.weights += 1,
            KAPSL_ALLOCATION_CLASS_WORKSPACE => ledger.workspace += 1,
            _ => ledger.other += 1,
        }
        ledger.live.insert(
            allocation_id,
            LiveAllocation {
                pointer: pointer as usize,
                bytes: request.bytes,
            },
        );
    }
    // SAFETY: allocation_out was checked and is writable for this call.
    unsafe {
        *allocation_out = KapslDeviceAllocationV1 {
            struct_size: std::mem::size_of::<KapslDeviceAllocationV1>() as u32,
            reserved: 0,
            allocation_id,
            device_ptr: pointer,
            granted_bytes: request.bytes,
        }
    };
    KAPSL_STATUS_OK
}

unsafe extern "C" fn gpu_allocate_unscoped(
    user_data: *mut c_void,
    _request: *const KapslDeviceAllocationRequestV1,
    _allocation_out: *mut KapslDeviceAllocationV1,
) -> i32 {
    if !user_data.is_null() {
        // SAFETY: the fixture keeps its GpuHost alive until shutdown returns.
        let host = unsafe { &*user_data.cast::<GpuHost>() };
        host.unscoped_allocations.fetch_add(1, Ordering::Relaxed);
    }
    // A scoped-allocator adapter must never use the unscoped entry point.
    KAPSL_STATUS_BACKEND_ERROR
}

unsafe extern "C" fn gpu_free(
    user_data: *mut c_void,
    allocation: *const KapslDeviceAllocationV1,
) -> i32 {
    if user_data.is_null() || allocation.is_null() {
        return KAPSL_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the fixture keeps its GpuHost alive until shutdown returns.
    let host = unsafe { &*user_data.cast::<GpuHost>() };
    // SAFETY: the adapter passes a readable allocation record for this call.
    let allocation = unsafe { *allocation };
    let Some(live) = host.ledger().live.remove(&allocation.allocation_id) else {
        host.rejected_frees.fetch_add(1, Ordering::Relaxed);
        return KAPSL_STATUS_INVALID_ARGUMENT;
    };
    if live.pointer != allocation.device_ptr as usize || live.bytes != allocation.granted_bytes {
        host.rejected_frees.fetch_add(1, Ordering::Relaxed);
        return KAPSL_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the pointer came from cudaMalloc and is freed exactly once.
    if unsafe { (host.cuda.free)(live.pointer as *mut c_void) } != 0 {
        return KAPSL_STATUS_BACKEND_ERROR;
    }
    KAPSL_STATUS_OK
}

unsafe extern "C" fn gpu_synchronize(user_data: *mut c_void, _device_id: u32) -> i32 {
    if user_data.is_null() {
        return KAPSL_STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: the fixture keeps its GpuHost alive until shutdown returns.
    let host = unsafe { &*user_data.cast::<GpuHost>() };
    // SAFETY: plain CUDA runtime call with no arguments.
    if unsafe { (host.cuda.synchronize)() } == 0 {
        KAPSL_STATUS_OK
    } else {
        KAPSL_STATUS_BACKEND_ERROR
    }
}

unsafe extern "C" fn gpu_log(_user_data: *mut c_void, level: u32, message: KapslSlice) {
    // SAFETY: the adapter keeps the message readable for this call.
    let message = unsafe { message.as_bytes() }
        .and_then(|bytes| std::str::from_utf8(bytes).ok())
        .unwrap_or("invalid log message");
    eprintln!("[ort-adapter level={level}] {message}");
}

struct GpuFixture {
    root: tempfile::TempDir,
    host: Box<GpuHost>,
    _table: Box<KapslBackendHostScopedAllocatorV1>,
    _profile: Vec<u8>,
    _manifest: Vec<u8>,
    _options: Vec<u8>,
    config: KapslBackendConfigV1,
}

impl GpuFixture {
    fn new(model_id: u32, peak_concurrency: u32, session_buckets: usize) -> Self {
        let root = tempfile::tempdir().unwrap();
        let entrypoint = root.path().join("libkapsl_backend_ort.test");
        std::fs::write(&entrypoint, b"test entrypoint").unwrap();
        let host = Box::new(GpuHost {
            cuda: Cuda::load(),
            next_id: AtomicU64::new(0),
            unscoped_allocations: AtomicUsize::new(0),
            rejected_frees: AtomicUsize::new(0),
            ledger: Mutex::new(Ledger::default()),
        });
        let base = KapslBackendHostV1 {
            struct_size: std::mem::size_of::<KapslBackendHostV1>() as u32,
            abi_version: KAPSL_BACKEND_ABI_VERSION,
            user_data: (host.as_ref() as *const GpuHost).cast_mut().cast(),
            log: Some(gpu_log),
            allocate_device: Some(gpu_allocate_unscoped),
            free_device: Some(gpu_free),
            synchronize_device: Some(gpu_synchronize),
        };
        let table = Box::new(KapslBackendHostScopedAllocatorV1::new(
            base,
            gpu_allocate_scoped,
        ));
        let profile = b"cuda12".to_vec();
        let manifest = serde_json::to_vec(&serde_json::json!({
            "project_name": "doubling",
            "framework": "onnx",
            "version": "0.1.0",
            "created_at": "2026-11-24T00:00:00Z",
            "model_file": "doubling.onnx",
            "format": "onnx",
            "model_type": "opaque",
            "task": "forward"
        }))
        .unwrap();
        let options = serde_json::to_vec(&serde_json::json!({
            "provider": "cuda",
            "accelerator_profile": "cuda",
            "pack_version": "ort-2.0.0-rc.11",
            "pack_root": root.path(),
            "entrypoint": entrypoint,
            "onnx_tuning": {
                "memory_pattern": true,
                "disable_cpu_mem_arena": false,
                "session_buckets": session_buckets,
                "bucket_dim_granularity": 64,
                "bucket_max_dims": 4,
                "peak_concurrency_hint": peak_concurrency
            }
        }))
        .unwrap();
        let config = KapslBackendConfigV1 {
            struct_size: std::mem::size_of::<KapslBackendConfigV1>() as u32,
            device_id: 0,
            // Distinct per test: a test that panics before shutdown leaves its
            // allocator client registered for the rest of the process.
            model_id,
            replica_id: 2,
            require_governed_device_memory: 1,
            reserved: 0,
            profile: KapslSlice::from_bytes(&profile),
            manifest_json: KapslSlice::from_bytes(&manifest),
            options_json: KapslSlice::from_bytes(&options),
            host: &table.base,
        };
        Self {
            root,
            host,
            _table: table,
            _profile: profile,
            _manifest: manifest,
            _options: options,
            config,
        }
    }

    fn initialize_and_load(&self) -> *mut c_void {
        let api = api();
        let mut handle = ptr::null_mut();
        let mut error = KapslOwnedBuffer::empty();
        // SAFETY: fixture storage outlives the backend handle.
        let status =
            unsafe { api.initialize.expect("initialize")(&self.config, &mut handle, &mut error) };
        assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));
        let model_path = self.root.path().join("doubling.onnx");
        std::fs::write(&model_path, doubling_matmul_onnx()).unwrap();
        let model_text = model_path.to_str().unwrap().as_bytes();
        let mut error = KapslOwnedBuffer::empty();
        // SAFETY: handle and model-path storage are live.
        let status = unsafe {
            api.load_model.expect("load")(handle, KapslSlice::from_bytes(model_text), &mut error)
        };
        assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));
        handle
    }
}

const HIDDEN: usize = 256;

/// `output[1,seq,256] = input[1,seq,256] x weight[256,256]` with
/// `weight = 2·I`, so the 256 KiB initializer is a real device upload and
/// the output is exactly twice the input.
fn doubling_matmul_onnx() -> Vec<u8> {
    let mut weight = vec![0.0_f32; HIDDEN * HIDDEN];
    for index in 0..HIDDEN {
        weight[index * HIDDEN + index] = 2.0;
    }
    let mut initializer = Vec::new();
    initializer.extend(varint_field(1, HIDDEN as u64));
    initializer.extend(varint_field(1, HIDDEN as u64));
    initializer.extend(varint_field(2, 1)); // TensorProto.FLOAT
    append_bytes(&mut initializer, 8, b"weight");
    let raw = weight
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    append_bytes(&mut initializer, 9, &raw);

    let value_info = |name: &[u8]| {
        let mut shape = Vec::new();
        append_bytes(&mut shape, 1, &varint_field(1, 1));
        let mut sequence = Vec::new();
        append_bytes(&mut sequence, 2, b"seq");
        append_bytes(&mut shape, 1, &sequence);
        append_bytes(&mut shape, 1, &varint_field(1, HIDDEN as u64));
        let mut tensor_type = varint_field(1, 1);
        append_bytes(&mut tensor_type, 2, &shape);
        let mut value_type = Vec::new();
        append_bytes(&mut value_type, 1, &tensor_type);
        let mut value = Vec::new();
        append_bytes(&mut value, 1, name);
        append_bytes(&mut value, 2, &value_type);
        value
    };
    let mut node = Vec::new();
    append_bytes(&mut node, 1, b"input");
    append_bytes(&mut node, 1, b"weight");
    append_bytes(&mut node, 2, b"output");
    append_bytes(&mut node, 4, b"MatMul");
    let mut graph = Vec::new();
    append_bytes(&mut graph, 1, &node);
    append_bytes(&mut graph, 2, b"doubling_matmul");
    append_bytes(&mut graph, 5, &initializer);
    append_bytes(&mut graph, 11, &value_info(b"input"));
    append_bytes(&mut graph, 12, &value_info(b"output"));

    let mut model = varint_field(1, 9);
    append_bytes(&mut model, 7, &graph);
    let mut opset = Vec::new();
    append_bytes(&mut opset, 1, b"");
    opset.extend(varint_field(2, 13));
    append_bytes(&mut model, 8, &opset);
    model
}

fn infer_doubling(handle: *mut c_void, request_id: u64, seq: i64) {
    let api = api();
    let values = (0..seq * HIDDEN as i64)
        .map(|value| (value % 251) as f32)
        .collect::<Vec<_>>();
    let input_bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let shape = [1_i64, seq, HIDDEN as i64];
    let input = KapslNamedTensorViewV1 {
        struct_size: std::mem::size_of::<KapslNamedTensorViewV1>() as u32,
        reserved: 0,
        name: KapslSlice::from_bytes(b"input"),
        tensor: KapslTensorViewV1 {
            struct_size: std::mem::size_of::<KapslTensorViewV1>() as u32,
            dtype: KAPSL_DTYPE_F32,
            memory_kind: KAPSL_MEMORY_HOST,
            flags: KAPSL_TENSOR_FLAG_CONTIGUOUS | KAPSL_TENSOR_FLAG_READ_ONLY,
            device_id: -1,
            rank: shape.len() as u32,
            shape: shape.as_ptr(),
            strides: ptr::null(),
            data: input_bytes.as_ptr().cast(),
            byte_len: input_bytes.len() as u64,
        },
    };
    let request = KapslInferenceRequestV1 {
        struct_size: std::mem::size_of::<KapslInferenceRequestV1>() as u32,
        wire_format: KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1,
        request_id,
        inputs: &input,
        input_count: 1,
        reserved: 0,
        metadata_json: KapslSlice::empty(),
        cancellation_context: ptr::null_mut(),
        is_cancelled: None,
    };
    let mut result = KapslInferenceResultV1::empty();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: request storage is live until infer returns.
    let status = unsafe { api.infer.expect("infer")(handle, &request, &mut result, &mut error) };
    assert_eq!(
        status,
        KAPSL_STATUS_OK,
        "seq {seq}: {}",
        take_error(api, error)
    );
    // SAFETY: result storage remains owned until this matching release.
    let output = unsafe { &*result.outputs };
    let output_bytes = unsafe {
        std::slice::from_raw_parts(
            output.tensor.data.cast::<u8>(),
            output.tensor.byte_len as usize,
        )
    };
    let doubled = output_bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(doubled.len(), values.len(), "seq {seq}");
    assert!(
        doubled
            .iter()
            .zip(&values)
            .all(|(out, value)| *out == value * 2.0),
        "seq {seq}: GPU output is not twice the input"
    );
    // SAFETY: this is the one release of an owned result.
    unsafe { api.release_result.expect("release")(handle, &mut result) };
}

fn unload_and_shutdown(handle: *mut c_void) {
    let api = api();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: the handle is live; unload drains its sessions.
    let status = unsafe { api.unload.expect("unload")(handle, &mut error) };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));
    // SAFETY: the handle is not used after shutdown.
    unsafe { api.shutdown.expect("shutdown")(handle) };
}

#[test]
#[ignore = "needs an NVIDIA GPU, libcudart.so.12 and ORT_LIB_LOCATION"]
fn cuda_input_lengths_share_one_device_weight_copy() {
    // kapsl-engine#250: with session_buckets=4, lengths 2, 100, 200 and 300
    // used to build a session (another device weight copy) per bucket.
    let fixture = GpuFixture::new(11, 1, 4);
    let handle = fixture.initialize_and_load();
    let weights_after_load = fixture.host.ledger().weights;
    assert!(
        weights_after_load > 0,
        "model load uploaded no weights through the governed allocator"
    );

    for (request_id, seq) in [(1_u64, 2_i64), (2, 100), (3, 200), (4, 300), (5, 60)] {
        infer_doubling(handle, request_id, seq);
    }

    let (weights, workspace, other) = {
        let ledger = fixture.host.ledger();
        (ledger.weights, ledger.workspace, ledger.other)
    };
    eprintln!("governed allocations: weights={weights} workspace={workspace} other={other}");
    assert_eq!(
        weights, weights_after_load,
        "a new input length uploaded another copy of the weights"
    );
    assert!(
        workspace > 0,
        "inference allocated no governed device workspace"
    );
    let metrics: EngineMetrics = json_report(api(), api().metrics.expect("metrics"), handle);
    assert_eq!(metrics.onnx_session_pool_total, 1);
    assert_eq!(metrics.error_rate, 0.0);

    unload_and_shutdown(handle);
    assert_eq!(fixture.host.unscoped_allocations.load(Ordering::Relaxed), 0);
    assert_eq!(fixture.host.rejected_frees.load(Ordering::Relaxed), 0);
    assert_eq!(
        fixture.host.ledger().live.len(),
        0,
        "device allocations outlived unload and shutdown"
    );
}

#[test]
#[ignore = "needs an NVIDIA GPU, libcudart.so.12 and ORT_LIB_LOCATION"]
fn cuda_session_pool_spawns_no_cpu_workers() {
    // Each CUDA session used to start ORT's default intra-op CPU pool, one
    // worker per core, multiplied by every prewarmed session.
    fn threads() -> usize {
        std::fs::read_dir("/proc/self/task").unwrap().count()
    }
    let cores = std::thread::available_parallelism().map_or(1, |cores| cores.get());
    let fixture = GpuFixture::new(12, 4, 1);
    let before = threads();
    let handle = fixture.initialize_and_load();
    let spawned = threads().saturating_sub(before);
    infer_doubling(handle, 1, 8);
    eprintln!("4 prewarmed CUDA sessions added {spawned} threads on a {cores}-core host");
    // The CUDA runtime starts a few service threads of its own; ORT's pools
    // would add (cores - 1) per session.
    assert!(
        spawned < cores.max(8),
        "4 CUDA sessions added {spawned} threads on a {cores}-core host"
    );
    unload_and_shutdown(handle);
}

fn api() -> &'static KapslBackendApiV1 {
    // SAFETY: the exported entrypoint returns a process-lifetime static table.
    unsafe { &*kapsl_backend_v1() }
}

fn take_error(api: &KapslBackendApiV1, buffer: KapslOwnedBuffer) -> String {
    String::from_utf8(take_buffer(api, buffer)).unwrap_or_else(|_| "non-UTF-8 error".to_string())
}

fn take_buffer(api: &KapslBackendApiV1, buffer: KapslOwnedBuffer) -> Vec<u8> {
    if buffer.ptr.is_null() {
        return Vec::new();
    }
    // SAFETY: the adapter retains this readable allocation until free_buffer.
    let bytes = unsafe { std::slice::from_raw_parts(buffer.ptr, buffer.len) }.to_vec();
    // SAFETY: this is the matching table function and the one release.
    unsafe { api.free_buffer.expect("free buffer")(buffer) };
    bytes
}

fn json_report<T: serde::de::DeserializeOwned>(
    api: &KapslBackendApiV1,
    report: KapslBackendJsonReportFn,
    handle: *mut c_void,
) -> T {
    let mut output = KapslOwnedBuffer::empty();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: handle is live and output slots are writable.
    let status = unsafe { report(handle, &mut output, &mut error) };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));
    serde_json::from_slice(&take_buffer(api, output)).unwrap()
}

fn varint_field(field: u64, value: u64) -> Vec<u8> {
    let mut output = Vec::new();
    append_varint(&mut output, field << 3);
    append_varint(&mut output, value);
    output
}

fn append_bytes(output: &mut Vec<u8>, field: u64, value: &[u8]) {
    append_varint(output, (field << 3) | 2);
    append_varint(output, value.len() as u64);
    output.extend_from_slice(value);
}

fn append_varint(output: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        output.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}
