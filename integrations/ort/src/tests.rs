use super::*;
use std::ptr;

const IDENTITY_EMBED_ONNX: &[u8] = &[
    0x08, 0x09, 0x3a, 0x8e, 0x01, 0x0a, 0x25, 0x0a, 0x06, 0x68, 0x69, 0x64, 0x64, 0x65, 0x6e, 0x12,
    0x11, 0x6c, 0x61, 0x73, 0x74, 0x5f, 0x68, 0x69, 0x64, 0x64, 0x65, 0x6e, 0x5f, 0x73, 0x74, 0x61,
    0x74, 0x65, 0x22, 0x08, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x12, 0x0e, 0x69, 0x64,
    0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 0x5f, 0x65, 0x6d, 0x62, 0x65, 0x64, 0x5a, 0x24, 0x0a, 0x06,
    0x68, 0x69, 0x64, 0x64, 0x65, 0x6e, 0x12, 0x1a, 0x0a, 0x18, 0x08, 0x01, 0x12, 0x14, 0x0a, 0x07,
    0x12, 0x05, 0x62, 0x61, 0x74, 0x63, 0x68, 0x0a, 0x05, 0x12, 0x03, 0x73, 0x65, 0x71, 0x0a, 0x02,
    0x08, 0x02, 0x62, 0x2f, 0x0a, 0x11, 0x6c, 0x61, 0x73, 0x74, 0x5f, 0x68, 0x69, 0x64, 0x64, 0x65,
    0x6e, 0x5f, 0x73, 0x74, 0x61, 0x74, 0x65, 0x12, 0x1a, 0x0a, 0x18, 0x08, 0x01, 0x12, 0x14, 0x0a,
    0x07, 0x12, 0x05, 0x62, 0x61, 0x74, 0x63, 0x68, 0x0a, 0x05, 0x12, 0x03, 0x73, 0x65, 0x71, 0x0a,
    0x02, 0x08, 0x02, 0x42, 0x04, 0x0a, 0x00, 0x10, 0x0d,
];

fn identity_onnx(shape: &[u64]) -> Vec<u8> {
    let mut tensor_shape = Vec::new();
    for dimension in shape {
        append_bytes(&mut tensor_shape, 1, &varint_field(1, *dimension));
    }
    let mut tensor_type = varint_field(1, 1); // TensorProto.FLOAT
    append_bytes(&mut tensor_type, 2, &tensor_shape);
    let mut value_type = Vec::new();
    append_bytes(&mut value_type, 1, &tensor_type);

    let value_info = |name: &[u8]| {
        let mut value = Vec::new();
        append_bytes(&mut value, 1, name);
        append_bytes(&mut value, 2, &value_type);
        value
    };
    let mut node = Vec::new();
    append_bytes(&mut node, 1, b"input");
    append_bytes(&mut node, 2, b"output");
    append_bytes(&mut node, 4, b"Identity");
    let mut graph = Vec::new();
    append_bytes(&mut graph, 1, &node);
    append_bytes(&mut graph, 2, b"identity");
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

#[test]
fn api_table_is_backend_abi_v1_compatible() {
    // SAFETY: the exported entrypoint returns a process-lifetime static table.
    let api = unsafe { &*kapsl_backend_v1() };
    assert!(api.is_compatible());
    assert!(api.has_required_functions());
    assert!(api.capabilities_are_consistent());
    assert_eq!(api.capabilities, CAPABILITIES);
    assert!(api.capabilities & KAPSL_BACKEND_CAP_BATCHING != 0);
    assert!(api.capabilities & KAPSL_BACKEND_CAP_CANCELLATION != 0);
    assert!(api.capabilities & KAPSL_BACKEND_CAP_CONCURRENT_INFERENCE != 0);
    assert!(api.infer_batch.is_some());
    assert!(api.cancel.is_some());
    assert!(api.release_batch_result.is_some());
}

#[test]
fn descriptor_is_backend_neutral_and_released_by_the_pack() {
    let api = api();
    let mut descriptor = KapslOwnedBuffer::empty();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: output buffers are writable for the duration of the call.
    let status = unsafe { api.describe.expect("describe")(&mut descriptor, &mut error) };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));
    let bytes = take_buffer(api, descriptor);
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["backend"], "onnx");
    assert_eq!(value["runtime"], "onnxruntime");
    assert_eq!(value["runtime_version"], "1.23.2");
    assert_eq!(value["binding"], "ort");
    assert_eq!(value["binding_version"], "2.0.0-rc.11");
    assert_eq!(value["phase"], "cpu-inflight-cancellation");
    assert_eq!(value["cancellation"], "ort-run-termination");
    assert_eq!(
        value["tasks"],
        serde_json::json!(["forward", "embed", "classify", "detect", "transcribe"])
    );
    assert_eq!(
        value["preprocessing"],
        serde_json::json!(["tensor", "vision", "audio"])
    );
    assert_eq!(value["governed_device_memory"], false);
}

#[test]
fn initialization_rejects_incoherent_and_generation_manifests() {
    let api = api();
    let invalid = InitFixture::with_manifest(
        0,
        1,
        serde_json::json!({
            "project_name": "invalid",
            "framework": "onnx",
            "version": "0.1.0",
            "created_at": "2026-09-01T00:00:00Z",
            "model_file": "invalid.onnx",
            "format": "onnx",
            "model_type": "opaque",
            "task": "classify"
        }),
    );
    let mut handle = ptr::null_mut();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: fixture storage outlives this synchronous initialization call.
    let status =
        unsafe { api.initialize.expect("initialize")(&invalid.config, &mut handle, &mut error) };
    assert_eq!(status, KAPSL_STATUS_INVALID_ARGUMENT);
    assert!(handle.is_null());
    assert!(take_error(api, error).contains("not valid for model_type"));

    let generation = InitFixture::with_task(0, 1, "generate", "causal-lm", None);
    let mut error = KapslOwnedBuffer::empty();
    let status =
        unsafe { api.initialize.expect("initialize")(&generation.config, &mut handle, &mut error) };
    assert_eq!(status, KAPSL_STATUS_UNSUPPORTED);
    assert!(handle.is_null());
    assert!(take_error(api, error).contains("generation profile"));

    let unknown_preprocessing = InitFixture::with_task(
        0,
        1,
        "forward",
        "opaque",
        Some(serde_json::json!({"preprocess": {"kind": "video"}})),
    );
    let mut error = KapslOwnedBuffer::empty();
    let status = unsafe {
        api.initialize.expect("initialize")(&unknown_preprocessing.config, &mut handle, &mut error)
    };
    assert_eq!(status, KAPSL_STATUS_INVALID_ARGUMENT);
    assert!(handle.is_null());
    assert!(take_error(api, error).contains("unknown metadata.preprocess kind"));
}

#[test]
fn cpu_adapter_rejects_governed_device_configuration() {
    let fixture = InitFixture::new(1);
    let api = api();
    let mut handle = ptr::null_mut();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: fixture storage outlives this synchronous initialization call.
    let status =
        unsafe { api.initialize.expect("initialize")(&fixture.config, &mut handle, &mut error) };
    assert_eq!(status, KAPSL_STATUS_INVALID_ARGUMENT);
    assert!(handle.is_null());
    assert!(take_error(api, error).contains("governed device memory"));
}

#[test]
fn cancellation_hook_is_idempotent_outside_an_active_request() {
    let fixture = InitFixture::new(0);
    let api = api();
    let mut handle = ptr::null_mut();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: fixture storage outlives initialization.
    let status =
        unsafe { api.initialize.expect("initialize")(&fixture.config, &mut handle, &mut error) };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));

    // SAFETY: the initialized handle remains live; completion/cancellation
    // races require unknown request IDs to be harmless and idempotent.
    assert_eq!(
        unsafe { api.cancel.expect("cancel")(handle, 9_999) },
        KAPSL_STATUS_OK
    );
    unsafe { api.shutdown.expect("shutdown")(handle) };
}

#[test]
fn real_ort_vision_preprocessing_runs_through_abi() {
    let api = api();
    let fixture = InitFixture::with_task(
        0,
        1,
        "forward",
        "opaque",
        Some(serde_json::json!({
            "preprocess": {
                "kind": "vision",
                "width": 2,
                "height": 1,
                "resize": "stretch",
                "layout": "nchw",
                "scale": 1.0
            }
        })),
    );
    let mut handle = ptr::null_mut();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: fixture storage outlives initialization.
    let status =
        unsafe { api.initialize.expect("initialize")(&fixture.config, &mut handle, &mut error) };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));

    let model_path = fixture.root.path().join("identity-vision.onnx");
    std::fs::write(&model_path, identity_onnx(&[1, 3, 1, 2])).unwrap();
    let model_text = model_path.to_str().unwrap().as_bytes();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: handle and model path storage remain live.
    let status = unsafe {
        api.load_model.expect("load")(handle, KapslSlice::from_bytes(model_text), &mut error)
    };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));

    let image = image::RgbImage::from_fn(2, 1, |x, _| {
        if x == 0 {
            image::Rgb([255, 0, 0])
        } else {
            image::Rgb([0, 255, 0])
        }
    });
    let mut encoded = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .unwrap();
    let encoded = encoded.into_inner();
    let shape = [encoded.len() as i64];
    let input = tensor_view("input", KAPSL_DTYPE_U8, &shape, &encoded);
    let request = KapslInferenceRequestV1 {
        struct_size: std::mem::size_of::<KapslInferenceRequestV1>() as u32,
        wire_format: KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1,
        request_id: 9,
        inputs: &input,
        input_count: 1,
        reserved: 0,
        metadata_json: KapslSlice::empty(),
        cancellation_context: ptr::null_mut(),
        is_cancelled: None,
    };
    let mut result = KapslInferenceResultV1::empty();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: request and result storage remain live for synchronous inference.
    let status = unsafe { api.infer.expect("infer")(handle, &request, &mut result, &mut error) };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));
    // SAFETY: adapter result storage remains live until release_result.
    let output = unsafe { &*result.outputs };
    let output_shape =
        unsafe { std::slice::from_raw_parts(output.tensor.shape, output.tensor.rank as usize) };
    let output_data = unsafe {
        std::slice::from_raw_parts(
            output.tensor.data.cast::<u8>(),
            output.tensor.byte_len as usize,
        )
    };
    let values = output_data
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(output_shape, [1, 3, 1, 2]);
    assert_eq!(values, [255.0, 0.0, 0.0, 255.0, 0.0, 0.0]);
    // SAFETY: these are the matching one-time lifecycle operations.
    unsafe {
        api.release_result.expect("release")(handle, &mut result);
        api.shutdown.expect("shutdown")(handle);
    }
}

#[test]
fn preprocessing_model_contract_mismatch_rolls_back_load() {
    let api = api();
    let fixture = InitFixture::with_task(
        0,
        1,
        "forward",
        "opaque",
        Some(serde_json::json!({
            "preprocess": {"kind": "vision", "width": 2, "height": 1}
        })),
    );
    let mut handle = ptr::null_mut();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: fixture storage outlives initialization.
    let status =
        unsafe { api.initialize.expect("initialize")(&fixture.config, &mut handle, &mut error) };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));

    let model_path = fixture.root.path().join("wrong-rank.onnx");
    std::fs::write(&model_path, IDENTITY_EMBED_ONNX).unwrap();
    let model_text = model_path.to_str().unwrap().as_bytes();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: handle and model path storage remain live.
    let status = unsafe {
        api.load_model.expect("load")(handle, KapslSlice::from_bytes(model_text), &mut error)
    };
    assert_eq!(status, KAPSL_STATUS_INVALID_ARGUMENT);
    assert!(take_error(api, error).contains("emits shape"));
    let memory: MemoryReport = json_report(api, api.actual_memory.expect("memory"), handle);
    assert!(memory.allocations.is_empty());
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: the handle remains valid after the transactional load failure.
    let status = unsafe { api.health_check.expect("health")(handle, &mut error) };
    assert_eq!(status, KAPSL_STATUS_BACKEND_ERROR);
    assert!(take_error(api, error).contains("not loaded"));
    // SAFETY: shutdown consumes the adapter after the failed load.
    unsafe { api.shutdown.expect("shutdown")(handle) };
}

#[test]
fn real_ort_audio_preprocessing_runs_through_abi_and_is_memory_planned() {
    let api = api();
    let fixture = InitFixture::with_task(
        0,
        1,
        "forward",
        "opaque",
        Some(serde_json::json!({
            "preprocess": {
                "kind": "audio",
                "sample_rate": 16000,
                "n_fft": 4,
                "hop_length": 2,
                "n_mels": 2,
                "f_min": 0.0,
                "f_max": 8000.0,
                "log": "none",
                "center": false,
                "layout": "time_mel"
            }
        })),
    );
    let mut handle = ptr::null_mut();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: fixture storage outlives initialization.
    let status =
        unsafe { api.initialize.expect("initialize")(&fixture.config, &mut handle, &mut error) };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));

    let model_path = fixture.root.path().join("identity-audio.onnx");
    std::fs::write(&model_path, IDENTITY_EMBED_ONNX).unwrap();
    let model_text = model_path.to_str().unwrap().as_bytes();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: handle and model path storage remain live.
    let status = unsafe {
        api.load_model.expect("load")(handle, KapslSlice::from_bytes(model_text), &mut error)
    };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));
    let resident: MemoryReport = json_report(api, api.actual_memory.expect("memory"), handle);
    assert!(resident.allocations.iter().any(|allocation| allocation
        .allocation_id
        .ends_with(":preprocessor")
        && allocation.bytes > 0));

    let shape = [6_i64];
    let input_bytes = [1.0_f32, 0.5, 0.0, -0.5, -1.0, 0.0]
        .iter()
        .flat_map(|sample| sample.to_ne_bytes())
        .collect::<Vec<_>>();
    let input = tensor_view("input", KAPSL_DTYPE_F32, &shape, &input_bytes);
    let request = KapslInferenceRequestV1 {
        struct_size: std::mem::size_of::<KapslInferenceRequestV1>() as u32,
        wire_format: KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1,
        request_id: 8,
        inputs: &input,
        input_count: 1,
        reserved: 0,
        metadata_json: KapslSlice::empty(),
        cancellation_context: ptr::null_mut(),
        is_cancelled: None,
    };

    let mut memory = KapslOwnedBuffer::empty();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: request storage remains live for the synchronous report.
    let status = unsafe {
        api.planned_request_memory.expect("request memory")(
            handle,
            &request,
            &mut memory,
            &mut error,
        )
    };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));
    let memory: MemoryReport = serde_json::from_slice(&take_buffer(api, memory)).unwrap();
    assert!(memory.allocations[0].bytes > input_bytes.len());

    let mut result = KapslInferenceResultV1::empty();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: request and output storage remain live for synchronous inference.
    let status = unsafe { api.infer.expect("infer")(handle, &request, &mut result, &mut error) };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));
    // SAFETY: adapter result storage remains live until release_result.
    let output = unsafe { &*result.outputs };
    let output_shape =
        unsafe { std::slice::from_raw_parts(output.tensor.shape, output.tensor.rank as usize) };
    let output_data = unsafe {
        std::slice::from_raw_parts(
            output.tensor.data.cast::<u8>(),
            output.tensor.byte_len as usize,
        )
    };
    assert_eq!(output_shape, [1, 2, 2]);
    assert_eq!(output_data.len(), 4 * std::mem::size_of::<f32>());
    assert!(output_data
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .all(f32::is_finite));
    // SAFETY: this is the matching one-time result release.
    unsafe { api.release_result.expect("release")(handle, &mut result) };

    let batch_shapes = [[6_i64], [6_i64]];
    let batch_data = [
        input_bytes,
        [0.0_f32, 0.25, 0.5, 0.75, 1.0, 0.0]
            .iter()
            .flat_map(|sample| sample.to_ne_bytes())
            .collect::<Vec<_>>(),
    ];
    let batch_inputs = std::array::from_fn::<_, 2, _>(|index| {
        tensor_view(
            "input",
            KAPSL_DTYPE_F32,
            &batch_shapes[index],
            &batch_data[index],
        )
    });
    let batch_requests = std::array::from_fn::<_, 2, _>(|index| KapslInferenceRequestV1 {
        struct_size: std::mem::size_of::<KapslInferenceRequestV1>() as u32,
        wire_format: KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1,
        request_id: 80 + index as u64,
        inputs: &batch_inputs[index],
        input_count: 1,
        reserved: 0,
        metadata_json: KapslSlice::empty(),
        cancellation_context: ptr::null_mut(),
        is_cancelled: None,
    });
    let batch = KapslInferenceBatchV1 {
        struct_size: std::mem::size_of::<KapslInferenceBatchV1>() as u32,
        request_count: batch_requests.len() as u32,
        requests: batch_requests.as_ptr(),
    };
    let mut batch_result = KapslInferenceBatchResultV1::empty();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: every batch input remains live through synchronous inference.
    let status = unsafe {
        api.infer_batch.expect("infer batch")(handle, &batch, &mut batch_result, &mut error)
    };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));
    assert_eq!(batch_result.result_count, 2);
    // SAFETY: batch result storage remains live until its matching release.
    let outputs = unsafe { std::slice::from_raw_parts(batch_result.results, 2) };
    for output in outputs {
        let tensor = unsafe { &*output.outputs };
        let shape =
            unsafe { std::slice::from_raw_parts(tensor.tensor.shape, tensor.tensor.rank as usize) };
        assert_eq!(shape, [1, 2, 2]);
    }
    // SAFETY: this is the matching one-time batch release.
    unsafe { api.release_batch_result.expect("release batch")(handle, &mut batch_result) };
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: the handle remains live through unload and the report below.
    let status = unsafe { api.unload.expect("unload")(handle, &mut error) };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));
    let after_unload: MemoryReport = json_report(api, api.actual_memory.expect("memory"), handle);
    assert_eq!(after_unload.allocations.len(), 1);
    assert!(after_unload.allocations[0]
        .allocation_id
        .ends_with(":preprocessor"));
    // SAFETY: shutdown consumes the remaining adapter state.
    unsafe { api.shutdown.expect("shutdown")(handle) };
}

#[test]
fn real_ort_cpu_session_round_trips_borrowed_tensor_views() {
    let api = api();
    let fixture = InitFixture::new(0);
    let mut handle = ptr::null_mut();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: fixture storage outlives initialization.
    let status =
        unsafe { api.initialize.expect("initialize")(&fixture.config, &mut handle, &mut error) };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));
    assert!(!handle.is_null());

    let model_path = fixture.root.path().join("identity.onnx");
    std::fs::write(&model_path, IDENTITY_EMBED_ONNX).unwrap();
    let model_text = model_path.to_str().unwrap().as_bytes();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: handle and model path storage are live.
    let status = unsafe {
        api.load_model.expect("load")(handle, KapslSlice::from_bytes(model_text), &mut error)
    };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));

    let shape = [1_i64, 2, 2];
    let values = [1.0_f32, 2.0, 3.0, 4.0];
    let input_bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
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
        request_id: 7,
        inputs: &input,
        input_count: 1,
        reserved: 0,
        metadata_json: KapslSlice::empty(),
        cancellation_context: ptr::null_mut(),
        is_cancelled: None,
    };
    let mut result = KapslInferenceResultV1::empty();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: all request storage remains live through synchronous inference.
    let status = unsafe { api.infer.expect("infer")(handle, &request, &mut result, &mut error) };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));
    assert_eq!(result.output_count, 1);
    assert!(!result.outputs.is_null());
    // SAFETY: result storage remains adapter-owned until release_result.
    let output = unsafe { &*result.outputs };
    assert_eq!(output.tensor.dtype, KAPSL_DTYPE_F32);
    assert_eq!(output.tensor.rank, 3);
    // SAFETY: output shape and data are retained by owner_context.
    let output_shape = unsafe { std::slice::from_raw_parts(output.tensor.shape, 3) };
    let output_data = unsafe {
        std::slice::from_raw_parts(
            output.tensor.data.cast::<u8>(),
            output.tensor.byte_len as usize,
        )
    };
    assert_eq!(output_shape, shape);
    assert_eq!(output_data, input_bytes);
    // SAFETY: release, unload, and shutdown each consume their matching live state.
    unsafe {
        api.release_result.expect("release result")(handle, &mut result);
    }
    assert!(result.owner_context.is_null());

    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: handle remains live until shutdown below.
    let status = unsafe { api.unload.expect("unload")(handle, &mut error) };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));
    unsafe { api.shutdown.expect("shutdown")(handle) };
}

#[test]
fn real_ort_batch_stacks_splits_postprocesses_and_reloads_with_pool_accounting() {
    let api = api();
    let fixture = InitFixture::with_task(
        0,
        2,
        "embed",
        "embedding",
        Some(serde_json::json!({"embed": {"normalize": false}})),
    );
    let mut handle = ptr::null_mut();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: fixture storage outlives initialization.
    let status =
        unsafe { api.initialize.expect("initialize")(&fixture.config, &mut handle, &mut error) };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));

    let model_path = fixture.root.path().join("identity-batch.onnx");
    std::fs::write(&model_path, IDENTITY_EMBED_ONNX).unwrap();
    let model_text = model_path.to_str().unwrap().as_bytes();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: handle and model-path storage are live.
    let status = unsafe {
        api.load_model.expect("load")(handle, KapslSlice::from_bytes(model_text), &mut error)
    };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));
    let info: kapsl_engine_api::EngineModelInfo =
        json_report(api, api.model_info.expect("model info"), handle);
    assert_eq!(info.output_names, ["embedding"]);
    assert_eq!(info.output_shapes, [vec![-1, 2]]);

    let shapes = [[1_i64, 2, 2], [1_i64, 2, 2]];
    let input_bytes = [
        [1.0_f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>(),
        [5.0_f32, 6.0, 7.0, 8.0]
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>(),
    ];
    let inputs = std::array::from_fn::<_, 2, _>(|index| KapslNamedTensorViewV1 {
        struct_size: std::mem::size_of::<KapslNamedTensorViewV1>() as u32,
        reserved: 0,
        name: KapslSlice::from_bytes(b"input"),
        tensor: KapslTensorViewV1 {
            struct_size: std::mem::size_of::<KapslTensorViewV1>() as u32,
            dtype: KAPSL_DTYPE_F32,
            memory_kind: KAPSL_MEMORY_HOST,
            flags: KAPSL_TENSOR_FLAG_CONTIGUOUS | KAPSL_TENSOR_FLAG_READ_ONLY,
            device_id: -1,
            rank: shapes[index].len() as u32,
            shape: shapes[index].as_ptr(),
            strides: ptr::null(),
            data: input_bytes[index].as_ptr().cast(),
            byte_len: input_bytes[index].len() as u64,
        },
    });
    let requests = std::array::from_fn::<_, 2, _>(|index| KapslInferenceRequestV1 {
        struct_size: std::mem::size_of::<KapslInferenceRequestV1>() as u32,
        wire_format: KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1,
        request_id: 100 + index as u64,
        inputs: &inputs[index],
        input_count: 1,
        reserved: 0,
        metadata_json: KapslSlice::empty(),
        cancellation_context: ptr::null_mut(),
        is_cancelled: None,
    });

    let mut request_memory = KapslOwnedBuffer::empty();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: request and output storage remain live for this report call.
    let status = unsafe {
        api.planned_request_memory.expect("request memory")(
            handle,
            &requests[0],
            &mut request_memory,
            &mut error,
        )
    };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));
    let request_memory: MemoryReport =
        serde_json::from_slice(&take_buffer(api, request_memory)).unwrap();
    assert_eq!(request_memory.allocations.len(), 1);
    assert_eq!(request_memory.allocations[0].bytes, input_bytes[0].len());

    let batch = KapslInferenceBatchV1 {
        struct_size: std::mem::size_of::<KapslInferenceBatchV1>() as u32,
        request_count: requests.len() as u32,
        requests: requests.as_ptr(),
    };
    let mut result = KapslInferenceBatchResultV1::empty();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: all request storage remains live through synchronous inference.
    let status =
        unsafe { api.infer_batch.expect("infer batch")(handle, &batch, &mut result, &mut error) };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));
    assert_eq!(result.result_count, 2);
    // SAFETY: the batch owner retains the result array until release below.
    let results = unsafe { std::slice::from_raw_parts(result.results, 2) };
    for (index, nested) in results.iter().enumerate() {
        assert_eq!(nested.output_count, 1);
        // SAFETY: each nested output is retained by the batch owner.
        let output = unsafe { &*nested.outputs };
        let output_name = unsafe { output.name.as_bytes() }.unwrap();
        let output_shape =
            unsafe { std::slice::from_raw_parts(output.tensor.shape, output.tensor.rank as usize) };
        let output_data = unsafe {
            std::slice::from_raw_parts(
                output.tensor.data.cast::<u8>(),
                output.tensor.byte_len as usize,
            )
        };
        let expected_values = if index == 0 {
            [2.0_f32, 3.0]
        } else {
            [6.0_f32, 7.0]
        };
        let expected_data = expected_values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        assert_eq!(output_name, b"embedding");
        assert_eq!(output_shape, [1, 2]);
        assert_eq!(output_data, expected_data);
    }
    // SAFETY: this is the matching one-time batch result release.
    unsafe { api.release_batch_result.expect("release batch")(handle, &mut result) };
    assert!(result.owner_context.is_null());

    let metrics: EngineMetrics = json_report(api, api.metrics.expect("metrics"), handle);
    assert_eq!(metrics.batch_size, 2);
    assert_eq!(metrics.onnx_session_pool_total, 2);
    assert_eq!(metrics.onnx_session_pool_idle, 2);
    let memory: MemoryReport = json_report(api, api.actual_memory.expect("memory"), handle);
    assert_eq!(memory.allocations.len(), 1);
    assert_eq!(memory.allocations[0].bytes, IDENTITY_EMBED_ONNX.len() * 2);

    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: lifecycle calls are serialized and handle remains live.
    let status = unsafe { api.unload.expect("unload")(handle, &mut error) };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));
    let memory: MemoryReport = json_report(api, api.actual_memory.expect("memory"), handle);
    assert!(memory.allocations.is_empty());

    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: an unloaded handle may load the same model again.
    let status = unsafe {
        api.load_model.expect("reload")(handle, KapslSlice::from_bytes(model_text), &mut error)
    };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));
    let mut error = KapslOwnedBuffer::empty();
    let status = unsafe { api.health_check.expect("health")(handle, &mut error) };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));
    unsafe { api.shutdown.expect("shutdown")(handle) };
}

#[test]
fn concurrent_abi_calls_share_the_bounded_session_pool() {
    let api = api();
    let fixture = InitFixture::with_peak_concurrency(0, 2);
    let mut handle = ptr::null_mut();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: fixture storage outlives initialization.
    let status =
        unsafe { api.initialize.expect("initialize")(&fixture.config, &mut handle, &mut error) };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));

    let model_path = fixture.root.path().join("identity-concurrent.onnx");
    std::fs::write(&model_path, IDENTITY_EMBED_ONNX).unwrap();
    let model_text = model_path.to_str().unwrap().as_bytes();
    let mut error = KapslOwnedBuffer::empty();
    // SAFETY: handle and model-path storage are live.
    let status = unsafe {
        api.load_model.expect("load")(handle, KapslSlice::from_bytes(model_text), &mut error)
    };
    assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));

    let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
    let handle_address = handle as usize;
    let threads = (0..8)
        .map(|worker| {
            let barrier = std::sync::Arc::clone(&barrier);
            let api_table = api;
            std::thread::spawn(move || {
                let api = api_table;
                let handle = handle_address as *mut c_void;
                let shape = [1_i64, 2, 2];
                let input_bytes = [worker as f32, 2.0, 3.0, 4.0]
                    .iter()
                    .flat_map(|value| value.to_ne_bytes())
                    .collect::<Vec<_>>();
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
                    request_id: 1_000 + worker,
                    inputs: &input,
                    input_count: 1,
                    reserved: 0,
                    metadata_json: KapslSlice::empty(),
                    cancellation_context: ptr::null_mut(),
                    is_cancelled: None,
                };
                barrier.wait();
                let mut result = KapslInferenceResultV1::empty();
                let mut error = KapslOwnedBuffer::empty();
                // SAFETY: this thread retains all request storage until infer returns.
                let status =
                    unsafe { api.infer.expect("infer")(handle, &request, &mut result, &mut error) };
                assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(api, error));
                // SAFETY: result storage remains owned until this matching release.
                let output = unsafe { &*result.outputs };
                let output_data = unsafe {
                    std::slice::from_raw_parts(
                        output.tensor.data.cast::<u8>(),
                        output.tensor.byte_len as usize,
                    )
                };
                assert_eq!(output_data, input_bytes);
                unsafe { api.release_result.expect("release")(handle, &mut result) };
            })
        })
        .collect::<Vec<_>>();
    for thread in threads {
        thread.join().unwrap();
    }

    let metrics: EngineMetrics = json_report(api, api.metrics.expect("metrics"), handle);
    assert_eq!(metrics.onnx_session_pool_total, 2);
    assert_eq!(metrics.onnx_session_pool_idle, 2);
    assert_eq!(metrics.error_rate, 0.0);
    unsafe { api.shutdown.expect("shutdown")(handle) };
}

fn api() -> &'static KapslBackendApiV1 {
    // SAFETY: the exported entrypoint returns a process-lifetime static table.
    unsafe { &*kapsl_backend_v1() }
}

fn tensor_view(name: &str, dtype: u32, shape: &[i64], data: &[u8]) -> KapslNamedTensorViewV1 {
    KapslNamedTensorViewV1 {
        struct_size: std::mem::size_of::<KapslNamedTensorViewV1>() as u32,
        reserved: 0,
        name: KapslSlice::from_bytes(name.as_bytes()),
        tensor: KapslTensorViewV1 {
            struct_size: std::mem::size_of::<KapslTensorViewV1>() as u32,
            dtype,
            memory_kind: KAPSL_MEMORY_HOST,
            flags: KAPSL_TENSOR_FLAG_CONTIGUOUS | KAPSL_TENSOR_FLAG_READ_ONLY,
            device_id: -1,
            rank: shape.len() as u32,
            shape: shape.as_ptr(),
            strides: ptr::null(),
            data: data.as_ptr().cast(),
            byte_len: data.len() as u64,
        },
    }
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

struct InitFixture {
    root: tempfile::TempDir,
    _profile: Vec<u8>,
    _manifest: Vec<u8>,
    _options: Vec<u8>,
    _host: Box<KapslBackendHostV1>,
    config: KapslBackendConfigV1,
}

impl InitFixture {
    fn new(require_governed_device_memory: u32) -> Self {
        Self::with_peak_concurrency(require_governed_device_memory, 1)
    }

    fn with_peak_concurrency(require_governed_device_memory: u32, peak_concurrency: u32) -> Self {
        Self::with_task(
            require_governed_device_memory,
            peak_concurrency,
            "forward",
            "opaque",
            None,
        )
    }

    fn with_task(
        require_governed_device_memory: u32,
        peak_concurrency: u32,
        task: &str,
        model_type: &str,
        metadata: Option<serde_json::Value>,
    ) -> Self {
        let mut manifest = serde_json::json!({
            "project_name": "identity",
            "framework": "onnx",
            "version": "0.1.0",
            "created_at": "2026-09-01T00:00:00Z",
            "model_file": "identity.onnx",
            "format": "onnx",
            "model_type": model_type,
            "task": task
        });
        if let Some(metadata) = metadata {
            manifest["metadata"] = metadata;
        }
        Self::with_manifest(require_governed_device_memory, peak_concurrency, manifest)
    }

    fn with_manifest(
        require_governed_device_memory: u32,
        peak_concurrency: u32,
        manifest: serde_json::Value,
    ) -> Self {
        let root = tempfile::tempdir().unwrap();
        let entrypoint = root.path().join("libkapsl_backend_ort.test");
        std::fs::write(&entrypoint, b"test entrypoint").unwrap();
        let profile = b"cpu".to_vec();
        let manifest = serde_json::to_vec(&manifest).unwrap();
        let options = serde_json::to_vec(&serde_json::json!({
            "provider": "CPU",
            "accelerator_profile": "cpu",
            "pack_version": "ort-2.0.0-rc.11",
            "pack_root": root.path(),
            "entrypoint": entrypoint,
            "onnx_tuning": {
                "memory_pattern": true,
                "disable_cpu_mem_arena": false,
                "session_buckets": 1,
                "bucket_dim_granularity": 64,
                "bucket_max_dims": 4,
                "peak_concurrency_hint": peak_concurrency
            }
        }))
        .unwrap();
        let host = Box::new(KapslBackendHostV1 {
            struct_size: std::mem::size_of::<KapslBackendHostV1>() as u32,
            abi_version: KAPSL_BACKEND_ABI_VERSION,
            user_data: ptr::null_mut(),
            log: None,
            allocate_device: None,
            free_device: None,
            synchronize_device: None,
        });
        let config = KapslBackendConfigV1 {
            struct_size: std::mem::size_of::<KapslBackendConfigV1>() as u32,
            device_id: 0,
            model_id: 11,
            replica_id: 2,
            require_governed_device_memory,
            reserved: 0,
            profile: KapslSlice::from_bytes(&profile),
            manifest_json: KapslSlice::from_bytes(&manifest),
            options_json: KapslSlice::from_bytes(&options),
            host: host.as_ref(),
        };
        Self {
            root,
            _profile: profile,
            _manifest: manifest,
            _options: options,
            _host: host,
            config,
        }
    }
}
