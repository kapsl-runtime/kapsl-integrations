//! Stateless task parity with the embedded ORT backend, through ABI v1 and a
//! real ORT session.
//!
//! Each test pins what the embedded backend answers for the same manifest and
//! request: detection boxes in original-image pixels, labeled JSON output,
//! strict `metadata.detect` keys, BGR channel order, undeclared request
//! inputs, and the conventional inputs embedded ORT fills in when a request
//! omits them.

use super::*;
use std::cell::Cell;

const ONNX_FLOAT: u64 = 1;
const ONNX_INT64: u64 = 7;

fn fixed_dimension(value: u64) -> Vec<u8> {
    varint_field(1, value)
}

fn named_dimension(parameter: &str) -> Vec<u8> {
    let mut dimension = Vec::new();
    append_bytes(&mut dimension, 2, parameter.as_bytes());
    dimension
}

fn value_info(name: &str, dtype: u64, dimensions: &[Vec<u8>]) -> Vec<u8> {
    let mut shape = Vec::new();
    for dimension in dimensions {
        append_bytes(&mut shape, 1, dimension);
    }
    let mut tensor_type = varint_field(1, dtype);
    append_bytes(&mut tensor_type, 2, &shape);
    let mut value_type = Vec::new();
    append_bytes(&mut value_type, 1, &tensor_type);
    let mut value = Vec::new();
    append_bytes(&mut value, 1, name.as_bytes());
    append_bytes(&mut value, 2, &value_type);
    value
}

fn float_initializer(name: &str, dimensions: &[u64], values: &[f32]) -> Vec<u8> {
    let mut packed = Vec::new();
    for dimension in dimensions {
        append_varint(&mut packed, *dimension);
    }
    let mut tensor = Vec::new();
    append_bytes(&mut tensor, 1, &packed);
    tensor.extend(varint_field(2, ONNX_FLOAT));
    append_bytes(&mut tensor, 8, name.as_bytes());
    append_bytes(&mut tensor, 9, &f32_bytes(values));
    tensor
}

fn node(
    operator: &str,
    inputs: &[&str],
    output: &str,
    integer_attribute: Option<(&str, u64)>,
) -> Vec<u8> {
    let mut node = Vec::new();
    for input in inputs {
        append_bytes(&mut node, 1, input.as_bytes());
    }
    append_bytes(&mut node, 2, output.as_bytes());
    append_bytes(&mut node, 4, operator.as_bytes());
    if let Some((name, value)) = integer_attribute {
        let mut attribute = Vec::new();
        append_bytes(&mut attribute, 1, name.as_bytes());
        attribute.extend(varint_field(3, value)); // AttributeProto.i
        attribute.extend(varint_field(20, 2)); // AttributeProto.type = INT
        append_bytes(&mut node, 5, &attribute);
    }
    node
}

fn onnx_model(graph: &[u8]) -> Vec<u8> {
    let mut model = varint_field(1, 9);
    append_bytes(&mut model, 7, graph);
    let mut opset = Vec::new();
    append_bytes(&mut opset, 1, b"");
    opset.extend(varint_field(2, 13));
    append_bytes(&mut model, 8, &opset);
    model
}

/// A detector whose `[1, anchors, 5]` prediction grid is a constant: the
/// real-ORT counterpart of the embedded tests' fixed-output engine. It takes
/// a `[1, 3, 100, 100]` image tensor and one class without objectness.
fn fixed_detector_onnx(rows: &[[f32; 5]]) -> Vec<u8> {
    let anchors = rows.len() as u64;
    let grid = rows.iter().flatten().copied().collect::<Vec<_>>();
    let mut graph = Vec::new();
    append_bytes(
        &mut graph,
        1,
        &node("Identity", &["grid"], "predictions", None),
    );
    append_bytes(&mut graph, 2, b"fixed-detector");
    append_bytes(
        &mut graph,
        5,
        &float_initializer("grid", &[1, anchors, 5], &grid),
    );
    append_bytes(
        &mut graph,
        11,
        &value_info(
            "images",
            ONNX_FLOAT,
            &[
                fixed_dimension(1),
                fixed_dimension(3),
                fixed_dimension(100),
                fixed_dimension(100),
            ],
        ),
    );
    append_bytes(
        &mut graph,
        12,
        &value_info(
            "predictions",
            ONNX_FLOAT,
            &[
                fixed_dimension(1),
                fixed_dimension(anchors),
                fixed_dimension(5),
            ],
        ),
    );
    onnx_model(&graph)
}

/// `output = tokens <operator> Cast(<second input>)` over `[batch, sequence]`:
/// a transformer-shaped two-input graph whose result exposes exactly what the
/// second, int64 input held.
fn two_input_onnx(second_input: &str, operator: &str) -> Vec<u8> {
    let shape = [named_dimension("batch"), named_dimension("sequence")];
    let mut graph = Vec::new();
    append_bytes(
        &mut graph,
        1,
        &node(
            "Cast",
            &[second_input],
            "second_as_float",
            Some(("to", ONNX_FLOAT)),
        ),
    );
    append_bytes(
        &mut graph,
        1,
        &node(operator, &["tokens", "second_as_float"], "output", None),
    );
    append_bytes(&mut graph, 2, b"two-input");
    append_bytes(&mut graph, 11, &value_info("tokens", ONNX_FLOAT, &shape));
    append_bytes(
        &mut graph,
        11,
        &value_info(second_input, ONNX_INT64, &shape),
    );
    append_bytes(&mut graph, 12, &value_info("output", ONNX_FLOAT, &shape));
    onnx_model(&graph)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn i64_bytes(values: &[i64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn png(image: image::RgbImage) -> Vec<u8> {
    let mut encoded = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .unwrap();
    encoded.into_inner()
}

/// One adapter-owned output tensor, copied out before its release.
struct Output {
    name: String,
    dtype: u32,
    shape: Vec<i64>,
    data: Vec<u8>,
}

impl Output {
    /// # Safety
    ///
    /// `view` must be an output the adapter still owns.
    unsafe fn copy(view: &KapslNamedTensorViewV1) -> Self {
        // SAFETY: the caller guarantees the adapter retains this output's
        // name, shape and data until its matching release.
        let (name, shape, data) = unsafe {
            (
                view.name.as_bytes().unwrap().to_vec(),
                std::slice::from_raw_parts(view.tensor.shape, view.tensor.rank as usize).to_vec(),
                if view.tensor.byte_len == 0 {
                    Vec::new()
                } else {
                    std::slice::from_raw_parts(
                        view.tensor.data.cast::<u8>(),
                        view.tensor.byte_len as usize,
                    )
                    .to_vec()
                },
            )
        };
        Self {
            name: String::from_utf8(name).unwrap(),
            dtype: view.tensor.dtype,
            shape,
            data,
        }
    }

    fn f32(&self) -> Vec<f32> {
        assert_eq!(self.dtype, KAPSL_DTYPE_F32);
        self.data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_ne_bytes(*bytes))
            .collect()
    }
}

/// An initialized adapter instance driven only through its ABI v1 table.
struct Adapter {
    api: &'static KapslBackendApiV1,
    handle: *mut c_void,
    next_request_id: Cell<u64>,
    fixture: InitFixture,
}

impl Adapter {
    fn initialize(
        task: &str,
        model_type: &str,
        metadata: Option<serde_json::Value>,
    ) -> Result<Self, (i32, String)> {
        let api = api();
        let fixture = InitFixture::with_task(0, 1, task, model_type, metadata);
        let mut handle = ptr::null_mut();
        let mut error = KapslOwnedBuffer::empty();
        // SAFETY: fixture storage outlives this synchronous initialization.
        let status = unsafe {
            api.initialize.expect("initialize")(&fixture.config, &mut handle, &mut error)
        };
        if status != KAPSL_STATUS_OK {
            assert!(handle.is_null());
            return Err((status, take_error(api, error)));
        }
        Ok(Self {
            api,
            handle,
            next_request_id: Cell::new(1),
            fixture,
        })
    }

    /// Initialize for a manifest and load `model`.
    fn serve(
        task: &str,
        model_type: &str,
        metadata: Option<serde_json::Value>,
        model: &[u8],
    ) -> Self {
        let adapter = Self::initialize(task, model_type, metadata)
            .unwrap_or_else(|(status, message)| panic!("initialize failed ({status}): {message}"));
        let model_path = adapter.fixture.root.path().join("model.onnx");
        std::fs::write(&model_path, model).unwrap();
        let model_text = model_path.to_str().unwrap().as_bytes();
        let mut error = KapslOwnedBuffer::empty();
        // SAFETY: the handle and model path storage remain live for this call.
        let status = unsafe {
            adapter.api.load_model.expect("load")(
                adapter.handle,
                KapslSlice::from_bytes(model_text),
                &mut error,
            )
        };
        assert_eq!(
            status,
            KAPSL_STATUS_OK,
            "{}",
            take_error(adapter.api, error)
        );
        adapter
    }

    fn request(&self, inputs: &[KapslNamedTensorViewV1]) -> KapslInferenceRequestV1 {
        let request_id = self.next_request_id.get();
        self.next_request_id.set(request_id + 1);
        KapslInferenceRequestV1 {
            struct_size: std::mem::size_of::<KapslInferenceRequestV1>() as u32,
            wire_format: KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1,
            request_id,
            inputs: inputs.as_ptr(),
            input_count: inputs.len() as u32,
            reserved: 0,
            metadata_json: KapslSlice::empty(),
            cancellation_context: ptr::null_mut(),
            is_cancelled: None,
        }
    }

    fn infer(&self, inputs: &[KapslNamedTensorViewV1]) -> Result<Output, (i32, String)> {
        let request = self.request(inputs);
        let mut result = KapslInferenceResultV1::empty();
        let mut error = KapslOwnedBuffer::empty();
        // SAFETY: request, input and result storage outlive this synchronous call.
        let status = unsafe {
            self.api.infer.expect("infer")(self.handle, &request, &mut result, &mut error)
        };
        if status != KAPSL_STATUS_OK {
            return Err((status, take_error(self.api, error)));
        }
        assert_eq!(result.output_count, 1);
        // SAFETY: the adapter retains the output until the release below,
        // which is its matching one-time release.
        unsafe {
            let output = Output::copy(&*result.outputs);
            self.api.release_result.expect("release")(self.handle, &mut result);
            Ok(output)
        }
    }

    fn infer_batch(
        &self,
        requests: &[&[KapslNamedTensorViewV1]],
    ) -> Result<Vec<Output>, (i32, String)> {
        let requests = requests
            .iter()
            .map(|inputs| self.request(inputs))
            .collect::<Vec<_>>();
        let batch = KapslInferenceBatchV1 {
            struct_size: std::mem::size_of::<KapslInferenceBatchV1>() as u32,
            request_count: requests.len() as u32,
            requests: requests.as_ptr(),
        };
        let mut result = KapslInferenceBatchResultV1::empty();
        let mut error = KapslOwnedBuffer::empty();
        // SAFETY: every request and input outlives this synchronous call.
        let status = unsafe {
            self.api.infer_batch.expect("infer batch")(self.handle, &batch, &mut result, &mut error)
        };
        if status != KAPSL_STATUS_OK {
            return Err((status, take_error(self.api, error)));
        }
        assert_eq!(result.result_count as usize, requests.len());
        // SAFETY: the batch owner retains every nested output until the
        // release below, which is its matching one-time release.
        unsafe {
            let outputs = std::slice::from_raw_parts(result.results, requests.len())
                .iter()
                .map(|nested| {
                    assert_eq!(nested.output_count, 1);
                    Output::copy(&*nested.outputs)
                })
                .collect();
            self.api.release_batch_result.expect("release batch")(self.handle, &mut result);
            Ok(outputs)
        }
    }

    /// Bytes of the single request-transient allocation the adapter plans.
    fn planned_request_bytes(&self, inputs: &[KapslNamedTensorViewV1]) -> usize {
        let request = self.request(inputs);
        let mut memory = KapslOwnedBuffer::empty();
        let mut error = KapslOwnedBuffer::empty();
        // SAFETY: request and output storage outlive this synchronous report.
        let status = unsafe {
            self.api.planned_request_memory.expect("request memory")(
                self.handle,
                &request,
                &mut memory,
                &mut error,
            )
        };
        assert_eq!(status, KAPSL_STATUS_OK, "{}", take_error(self.api, error));
        let report: MemoryReport = serde_json::from_slice(&take_buffer(self.api, memory)).unwrap();
        assert_eq!(report.allocations.len(), 1);
        report.allocations[0].bytes
    }

    fn model_info(&self) -> kapsl_engine_api::EngineModelInfo {
        json_report(
            self.api,
            self.api.model_info.expect("model info"),
            self.handle,
        )
    }
}

impl Drop for Adapter {
    fn drop(&mut self) {
        // SAFETY: initialization returned this handle and nothing else
        // consumes it; the fixture it borrows is dropped afterwards.
        unsafe { self.api.shutdown.expect("shutdown")(self.handle) };
    }
}

fn letterbox_100() -> serde_json::Value {
    serde_json::json!({"kind": "vision", "width": 100, "height": 100, "resize": "letterbox"})
}

fn image_view<'a>(shape: &'a [i64; 1], encoded: &'a [u8]) -> KapslNamedTensorViewV1 {
    tensor_view("input", KAPSL_DTYPE_U8, shape, encoded)
}

#[test]
fn detection_reports_original_image_boxes_through_real_ort() {
    // Embedded `letterbox_boxes_map_back_to_original_image_pixels` and
    // `batch_maps_each_request_with_its_own_image_size`, on the default
    // `metadata.detect` coordinates.
    let adapter = Adapter::serve(
        "detect",
        "detector",
        Some(serde_json::json!({
            "detect": {"num_classes": 1, "box_format": "xyxy", "objectness": false},
            "preprocess": letterbox_100()
        })),
        &fixed_detector_onnx(&[[10.0, 35.0, 60.0, 60.0, 0.9]]),
    );
    let info = adapter.model_info();
    assert_eq!(info.output_names, ["detections"]);
    assert_eq!(info.output_shapes, [vec![-1, 6]]);
    assert_eq!(info.output_dtypes, ["float32"]);

    let wide = png(image::RgbImage::new(200, 100));
    let tall = png(image::RgbImage::new(100, 200));
    let (wide_shape, tall_shape) = ([wide.len() as i64], [tall.len() as i64]);
    let wide_request = [image_view(&wide_shape, &wide)];
    let tall_request = [image_view(&tall_shape, &tall)];

    // 200x100 into 100x100: scale 0.5, pad_y 25 -> (20,20)-(120,70).
    let output = adapter.infer(&wide_request).unwrap();
    assert_eq!(output.name, "detections");
    assert_eq!(output.shape, [1, 6]);
    assert_eq!(output.f32(), [20.0, 20.0, 120.0, 70.0, 0.9, 0.0]);

    // 100x200: scale 0.5, pad_x 25 -> x1 clips to 0, (0,70)-(70,120).
    let outputs = adapter
        .infer_batch(&[&wide_request, &tall_request])
        .unwrap();
    assert_eq!(outputs[0].f32(), [20.0, 20.0, 120.0, 70.0, 0.9, 0.0]);
    assert_eq!(outputs[1].f32(), [0.0, 70.0, 70.0, 120.0, 0.9, 0.0]);
}

#[test]
fn detection_model_coordinates_and_labeled_json_through_real_ort() {
    let serve = |detect: serde_json::Value| {
        Adapter::serve(
            "detect",
            "detector",
            Some(serde_json::json!({"detect": detect, "preprocess": letterbox_100()})),
            &fixed_detector_onnx(&[[-5.0, 35.0, 60.0, 130.0, 0.9]]),
        )
    };
    let encoded = png(image::RgbImage::new(200, 100));
    let shape = [encoded.len() as i64];
    let request = [image_view(&shape, &encoded)];

    // Embedded `model_coordinates_flag_keeps_model_space_and_clips_to_the_input`.
    let model_space = serve(serde_json::json!({
        "num_classes": 1,
        "box_format": "xyxy",
        "objectness": false,
        "coordinates": "model"
    }));
    assert_eq!(
        model_space.infer(&request).unwrap().f32(),
        [0.0, 35.0, 60.0, 100.0, 0.9, 0.0]
    );

    // The same box in original pixels is (-10,20)-(120,210), clipped to the
    // 200x100 image, and `json` carries the class label as UTF-8.
    let labeled = serve(serde_json::json!({
        "num_classes": 1,
        "box_format": "xyxy",
        "objectness": false,
        "labels": ["thing"],
        "output_format": "json"
    }));
    let info = labeled.model_info();
    assert_eq!(info.output_names, ["detections"]);
    assert_eq!(info.output_shapes, [vec![-1]]);
    assert_eq!(info.output_dtypes, ["string"]);
    let output = labeled.infer(&request).unwrap();
    assert_eq!(output.name, "detections");
    assert_eq!(output.dtype, KAPSL_DTYPE_UTF8);
    assert_eq!(output.shape, [output.data.len() as i64]);
    assert_eq!(
        std::str::from_utf8(&output.data).unwrap(),
        r#"[{"box":[0.0,20.0,120.0,100.0],"score":0.9,"class_id":0,"label":"thing"}]"#
    );
}

#[test]
fn detection_configuration_mistakes_fail_initialization() {
    for (detect, expected) in [
        (
            serde_json::json!({"num_classes": 1, "lables": ["thing"]}),
            "unknown field `lables`",
        ),
        (
            serde_json::json!({"num_classes": 1, "output_format": "json"}),
            "requires `labels`",
        ),
        (
            serde_json::json!({"num_classes": 2, "labels": ["only-one"]}),
            "labels has 1 entries but num_classes is 2",
        ),
    ] {
        let (status, message) = Adapter::initialize(
            "detect",
            "detector",
            Some(serde_json::json!({ "detect": detect })),
        )
        .err()
        .unwrap();
        assert_eq!(status, KAPSL_STATUS_INVALID_ARGUMENT, "{message}");
        assert!(message.contains(expected), "{message}");
    }
}

#[test]
fn vision_bgr_channel_order_reaches_real_ort() {
    let adapter = Adapter::serve(
        "forward",
        "opaque",
        Some(serde_json::json!({
            "preprocess": {
                "kind": "vision",
                "width": 2,
                "height": 1,
                "resize": "stretch",
                "layout": "nchw",
                "scale": 1.0,
                "channel_order": "bgr"
            }
        })),
        &identity_onnx(&[1, 3, 1, 2]),
    );
    let encoded = png(image::RgbImage::from_fn(2, 1, |x, _| {
        if x == 0 {
            image::Rgb([255, 0, 0])
        } else {
            image::Rgb([0, 255, 0])
        }
    }));
    let shape = [encoded.len() as i64];
    let output = adapter.infer(&[image_view(&shape, &encoded)]).unwrap();
    assert_eq!(output.shape, [1, 3, 1, 2]);
    // Planes are B, G, R; `rgb` emits [255, 0, 0, 255, 0, 0] for this image.
    assert_eq!(output.f32(), [0.0, 0.0, 0.0, 255.0, 255.0, 0.0]);
}

#[test]
fn undeclared_request_inputs_are_ignored_by_ort_and_still_pool_embeddings() {
    // The graph has one input. Embedded ORT does not send the two extra
    // tensors to the session, yet masked-mean pooling reads `attention_mask`.
    let adapter = Adapter::serve(
        "embed",
        "embedding",
        Some(serde_json::json!({"embed": {"normalize": false}})),
        IDENTITY_EMBED_ONNX,
    );
    let hidden_bytes = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let mask_bytes = i64_bytes(&[1, 0]);
    let type_bytes = i64_bytes(&[0, 0]);
    let (hidden_shape, token_shape) = ([1_i64, 2, 2], [1_i64, 2]);
    let hidden = tensor_view("input", KAPSL_DTYPE_F32, &hidden_shape, &hidden_bytes);
    let mask = tensor_view("attention_mask", KAPSL_DTYPE_I64, &token_shape, &mask_bytes);
    let types = tensor_view("token_type_ids", KAPSL_DTYPE_I64, &token_shape, &type_bytes);

    let pooled = adapter.infer(&[hidden, mask, types]).unwrap();
    assert_eq!(pooled.name, "embedding");
    assert_eq!(pooled.shape, [1, 2]);
    assert_eq!(pooled.f32(), [1.0, 2.0]);
    assert_eq!(adapter.infer(&[hidden]).unwrap().f32(), [2.0, 3.0]);
    assert_eq!(
        adapter.planned_request_bytes(&[hidden, mask, types]),
        hidden_bytes.len() + mask_bytes.len() + type_bytes.len()
    );

    let outputs = adapter
        .infer_batch(&[&[hidden, types, mask], &[hidden]])
        .unwrap();
    assert_eq!(outputs[0].f32(), [1.0, 2.0]);
    assert_eq!(outputs[1].f32(), [2.0, 3.0]);
}

#[test]
fn missing_attention_mask_and_position_ids_are_filled_like_embedded_ort() {
    let token_values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let token_bytes = f32_bytes(&token_values);
    let mask_bytes = i64_bytes(&[1, 0, 1, 0, 1, 0]);
    let type_bytes = i64_bytes(&[0; 6]);
    let shape = [2_i64, 3];
    let tokens = tensor_view("input", KAPSL_DTYPE_F32, &shape, &token_bytes);
    let mask = tensor_view("attention_mask", KAPSL_DTYPE_I64, &shape, &mask_bytes);
    let types = tensor_view("token_type_ids", KAPSL_DTYPE_I64, &shape, &type_bytes);
    let alternating = [1.0, 0.0, 3.0, 0.0, 5.0, 0.0];

    // output = tokens * attention_mask. A request without the mask gets an
    // all-ones int64 [batch, sequence] mask; a supplied one is honoured.
    let masked = Adapter::serve(
        "forward",
        "opaque",
        None,
        &two_input_onnx("attention_mask", "Mul"),
    );
    assert_eq!(
        masked.model_info().input_names,
        ["tokens", "attention_mask"]
    );
    assert_eq!(masked.infer(&[tokens]).unwrap().f32(), token_values);
    assert_eq!(masked.infer(&[tokens, mask]).unwrap().f32(), alternating);
    // An undeclared tensor is ignored on a multi-input graph as well.
    assert_eq!(masked.infer(&[tokens, types]).unwrap().f32(), token_values);
    assert_eq!(
        masked.infer(&[tokens, types, mask]).unwrap().f32(),
        alternating
    );
    let outputs = masked.infer_batch(&[&[tokens], &[tokens, mask]]).unwrap();
    assert_eq!(outputs[0].f32(), token_values);
    assert_eq!(outputs[1].f32(), alternating);

    // The synthesized mask is request-transient memory, so it is planned.
    let synthesized = 2 * 3 * std::mem::size_of::<i64>();
    assert_eq!(
        masked.planned_request_bytes(&[tokens]),
        token_bytes.len() + synthesized
    );
    assert_eq!(
        masked.planned_request_bytes(&[tokens, mask]),
        token_bytes.len() + mask_bytes.len()
    );
    assert_eq!(
        masked.planned_request_bytes(&[tokens, types]),
        token_bytes.len() + type_bytes.len() + synthesized
    );

    // output = tokens + position_ids, which count up within each batch row.
    let positioned = Adapter::serve(
        "forward",
        "opaque",
        None,
        &two_input_onnx("position_ids", "Add"),
    );
    assert_eq!(
        positioned.infer(&[tokens]).unwrap().f32(),
        [1.0, 3.0, 5.0, 4.0, 6.0, 8.0]
    );
}

#[test]
fn inputs_the_adapter_cannot_synthesize_remain_invalid_requests() {
    // output = tokens + token_type_ids. Embedded ORT skips this input and
    // fails inside the run; the adapter rejects the request before it.
    let adapter = Adapter::serve(
        "forward",
        "opaque",
        None,
        &two_input_onnx("token_type_ids", "Add"),
    );
    let token_bytes = f32_bytes(&[1.0, 2.0, 3.0]);
    let type_bytes = i64_bytes(&[10, 20, 30]);
    let shape = [1_i64, 3];
    let tokens = tensor_view("input", KAPSL_DTYPE_F32, &shape, &token_bytes);
    let types = tensor_view("token_type_ids", KAPSL_DTYPE_I64, &shape, &type_bytes);

    let (status, message) = adapter.infer(&[tokens]).err().unwrap();
    assert_eq!(status, KAPSL_STATUS_INVALID_ARGUMENT, "{message}");
    assert!(
        message.contains("missing input `token_type_ids`"),
        "{message}"
    );
    assert_eq!(adapter.planned_request_bytes(&[tokens]), token_bytes.len());
    // The rejected request leaves the session usable.
    assert_eq!(
        adapter.infer(&[tokens, types]).unwrap().f32(),
        [11.0, 22.0, 33.0]
    );
}
