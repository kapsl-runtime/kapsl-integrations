//! Detection postprocessing parity tests.
//!
//! Ported from the embedded ORT backend's
//! `kapsl-backends/src/onnx/detect_tests.rs` with the same numeric cases, so
//! one manifest and one request are held to the same answer on both routes.
//! Where the embedded tests wrap a fixed-output engine, these drive the
//! adapter's own manifest parsing, vision preprocessing and postprocessing
//! around a fixed raw model output.

use super::*;
use crate::preprocess::InputPreprocessor;
use kapsl_backend_abi::{KAPSL_DTYPE_U8, KAPSL_STATUS_INVALID_ARGUMENT};

fn cfg(num_classes: usize) -> DetectConfig {
    DetectConfig {
        num_classes,
        score_threshold: 0.25,
        iou_threshold: 0.45,
        max_detections: 300,
        box_format: BoxFormat::Xyxy,
        objectness: false,
        transposed: false,
        class_agnostic: false,
        labels: None,
        coordinates: Coordinates::default(),
        output_format: OutputFormat::default(),
    }
}

/// Build a raw anchor-major `[anchors, channels]` float32 output from rows.
fn raw_output(rows: &[Vec<f32>]) -> OwnedTensor {
    let flat = rows.iter().flatten().copied().collect();
    f32_tensor("raw", vec![rows.len() as i64, rows[0].len() as i64], flat)
}

fn decode_rows(output: &OwnedTensor) -> Vec<[f32; 6]> {
    assert_eq!(output.name, "detections");
    assert_eq!(output.dtype, KAPSL_DTYPE_F32);
    assert_eq!(output.shape[1], 6);
    let rows = parse_f32(&output.data).as_chunks::<6>().0.to_vec();
    assert_eq!(output.shape[0], rows.len() as i64);
    rows
}

/// Decode `rows` with boxes left in model-input space.
fn detect_from_output(rows: &[Vec<f32>], config: &DetectConfig) -> FfiResult<OwnedTensor> {
    detect(raw_output(rows), config, None)
}

fn names(count: usize) -> Vec<String> {
    (0..count).map(|index| format!("class{index}")).collect()
}

#[test]
fn thresholds_out_low_scores() {
    // 2 classes, xyxy, no objectness. One strong box, one below threshold.
    let output = detect_from_output(
        &[
            vec![0.0, 0.0, 10.0, 10.0, 0.9, 0.1],   // class 0 @ 0.9
            vec![50.0, 50.0, 60.0, 60.0, 0.1, 0.2], // best 0.2 < 0.25 -> dropped
        ],
        &cfg(2),
    )
    .unwrap();
    assert_eq!(decode_rows(&output), vec![[0.0, 0.0, 10.0, 10.0, 0.9, 0.0]]);
}

#[test]
fn nms_suppresses_overlapping_same_class() {
    // Two near-identical class-0 boxes -> only the higher-scoring one survives.
    let output = detect_from_output(
        &[
            vec![0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
            vec![0.5, 0.5, 10.5, 10.5, 0.8, 0.0], // IoU high with the first
        ],
        &cfg(2),
    )
    .unwrap();
    let detections = decode_rows(&output);
    assert_eq!(detections.len(), 1);
    assert_eq!(detections[0][4], 0.9);
}

#[test]
fn nms_keeps_overlapping_different_classes() {
    // Same location, different class -> per-class NMS keeps both.
    let output = detect_from_output(
        &[
            vec![0.0, 0.0, 10.0, 10.0, 0.9, 0.1], // class 0
            vec![0.5, 0.5, 10.5, 10.5, 0.1, 0.8], // class 1
        ],
        &cfg(2),
    )
    .unwrap();
    let mut detections = decode_rows(&output);
    detections.sort_by(|left, right| left[5].partial_cmp(&right[5]).unwrap());
    assert_eq!(detections.len(), 2);
    assert_eq!(detections[0][5], 0.0);
    assert_eq!(detections[1][5], 1.0);
}

#[test]
fn class_agnostic_suppresses_across_classes() {
    let mut config = cfg(2);
    config.class_agnostic = true;
    let output = detect_from_output(
        &[
            vec![0.0, 0.0, 10.0, 10.0, 0.9, 0.1],
            vec![0.5, 0.5, 10.5, 10.5, 0.1, 0.8],
        ],
        &config,
    )
    .unwrap();
    assert_eq!(decode_rows(&output).len(), 1);
}

#[test]
fn objectness_multiplies_class_score() {
    // YOLOv5 style: channels = 4 box + 1 obj + 2 classes.
    let mut config = cfg(2);
    config.objectness = true;
    // obj=0.5, class0=0.4 -> score 0.2 (< 0.25) dropped; obj=0.9, class1=0.9 -> 0.81 kept
    let output = detect_from_output(
        &[
            vec![0.0, 0.0, 10.0, 10.0, 0.5, 0.4, 0.1],
            vec![20.0, 20.0, 30.0, 30.0, 0.9, 0.1, 0.9],
        ],
        &config,
    )
    .unwrap();
    let detections = decode_rows(&output);
    assert_eq!(detections.len(), 1);
    assert_eq!(detections[0][5], 1.0);
    assert!((detections[0][4] - 0.81).abs() < 1e-6);
}

#[test]
fn xywh_converts_to_corners() {
    let mut config = cfg(1);
    config.box_format = BoxFormat::Xywh;
    // center (10,10), size 4x6 -> x1=8,y1=7,x2=12,y2=13
    let output = detect_from_output(&[vec![10.0, 10.0, 4.0, 6.0, 0.9]], &config).unwrap();
    let detection = decode_rows(&output)[0];
    assert_eq!(detection[..4], [8.0, 7.0, 12.0, 13.0]);
}

#[test]
fn transposed_layout_channel_major() {
    // YOLOv8 style [channels, anchors]: 4 box + 2 classes, 2 anchors.
    let mut config = cfg(2);
    config.transposed = true;
    // Anchor0: box(0,0,10,10) cls(0.9,0.1); Anchor1: box(20,20,30,30) cls(0.1,0.8).
    // Channel-major flat: [x1's, y1's, x2's, y2's, cls0's, cls1's]
    let flat = vec![
        0.0, 20.0, // channel 0 (x1) for anchors 0,1
        0.0, 20.0, // channel 1 (y1)
        10.0, 30.0, // channel 2 (x2)
        10.0, 30.0, // channel 3 (y2)
        0.9, 0.1, // channel 4 (class 0)
        0.1, 0.8, // channel 5 (class 1)
    ];
    let output = detect(f32_tensor("raw", vec![6, 2], flat), &config, None).unwrap();
    let mut detections = decode_rows(&output);
    detections.sort_by(|left, right| left[5].partial_cmp(&right[5]).unwrap());
    assert_eq!(detections.len(), 2);
    assert_eq!(detections[0][..4], [0.0, 0.0, 10.0, 10.0]);
    assert_eq!(detections[0][5], 0.0);
    assert_eq!(detections[1][..4], [20.0, 20.0, 30.0, 30.0]);
    assert_eq!(detections[1][5], 1.0);
}

#[test]
fn channel_count_mismatch_errors() {
    // Config expects 4+0+2=6 channels; give 5.
    assert!(detect_from_output(&[vec![0.0, 0.0, 10.0, 10.0, 0.9]], &cfg(2)).is_err());
}

#[test]
fn max_detections_caps_output() {
    let mut config = cfg(1);
    config.max_detections = 2;
    // Four non-overlapping high-score boxes; cap to 2 highest.
    let output = detect_from_output(
        &[
            vec![0.0, 0.0, 5.0, 5.0, 0.5],
            vec![10.0, 10.0, 15.0, 15.0, 0.9],
            vec![20.0, 20.0, 25.0, 25.0, 0.7],
            vec![30.0, 30.0, 35.0, 35.0, 0.6],
        ],
        &config,
    )
    .unwrap();
    let detections = decode_rows(&output);
    assert_eq!(detections.len(), 2);
    assert_eq!(detections[0][4], 0.9);
    assert_eq!(detections[1][4], 0.7);
}

#[test]
fn empty_when_all_below_threshold() {
    let output = detect_from_output(&[vec![0.0, 0.0, 10.0, 10.0, 0.01]], &cfg(1)).unwrap();
    assert_eq!(output.shape, vec![0, 6]);
    assert!(decode_rows(&output).is_empty());
}

// ---- labels, strict config, output format ----

#[test]
fn labels_length_must_match_num_classes() {
    let mut config = cfg(3);
    config.labels = Some(names(2));
    let error = config.validate().unwrap_err();
    assert_eq!(error.0, KAPSL_STATUS_INVALID_ARGUMENT);
    assert!(
        error
            .1
            .contains("labels has 2 entries but num_classes is 3"),
        "{}",
        error.1
    );
    config.labels = Some(Vec::new());
    assert!(
        config.validate().is_err(),
        "empty labels array is a length mismatch"
    );
    config.labels = Some(names(3));
    assert!(config.validate().is_ok());
}

#[test]
fn unknown_detect_keys_are_rejected() {
    // A typo must fail loudly instead of being ignored.
    let error = serde_yaml::from_str::<DetectConfig>("num_classes: 2\nlables: [a, b]").unwrap_err();
    assert!(
        error.to_string().contains("unknown field `lables`"),
        "{error}"
    );
    assert!(serde_yaml::from_str::<DetectConfig>("num_classes: 2\nclass_agnostic: true").is_ok());
}

#[test]
fn coordinates_parse_and_default_to_original() {
    let config: DetectConfig = serde_yaml::from_str("num_classes: 1").unwrap();
    assert_eq!(config.coordinates, Coordinates::Original);
    let config: DetectConfig = serde_yaml::from_str("num_classes: 1\ncoordinates: model").unwrap();
    assert_eq!(config.coordinates, Coordinates::Model);
    assert!(serde_yaml::from_str::<DetectConfig>("num_classes: 1\ncoordinates: bogus").is_err());
}

#[test]
fn json_output_format_includes_class_id_and_label() {
    let mut config = cfg(2);
    config.output_format = OutputFormat::Json;
    config.labels = Some(vec!["cat".into(), "dog \"rex\"".into()]);
    let output = detect_from_output(
        &[
            vec![0.0, 0.0, 10.0, 10.0, 0.1, 0.9],
            vec![50.0, 50.0, 60.0, 60.0, 0.8, 0.1],
        ],
        &config,
    )
    .unwrap();
    assert_eq!(output.name, "detections");
    assert_eq!(output.dtype, KAPSL_DTYPE_UTF8);
    assert_eq!(output.shape, vec![output.data.len() as i64]);
    let value: serde_json::Value = serde_json::from_slice(&output.data).unwrap();
    let rows = value.as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["class_id"], 1);
    assert_eq!(rows[0]["label"], "dog \"rex\""); // escaping survives
    assert_eq!(rows[0]["box"], serde_json::json!([0.0, 0.0, 10.0, 10.0]));
    assert!((rows[0]["score"].as_f64().unwrap() - 0.9).abs() < 1e-6);
    assert_eq!(rows[1]["class_id"], 0);
    assert_eq!(rows[1]["label"], "cat");
}

#[test]
fn empty_detections_in_json_format_encode_as_empty_json_array() {
    let mut config = cfg(1);
    config.output_format = OutputFormat::Json;
    config.labels = Some(names(1));
    let output = detect_from_output(&[vec![0.0, 0.0, 10.0, 10.0, 0.01]], &config).unwrap();
    assert_eq!(output.dtype, KAPSL_DTYPE_UTF8);
    assert_eq!(output.data, b"[]");
    assert_eq!(output.shape, vec![2]);
}

#[test]
fn tensor_output_format_is_default_and_ignores_labels() {
    let config: DetectConfig = serde_yaml::from_str("num_classes: 2").unwrap();
    assert_eq!(config.output_format, OutputFormat::Tensor);

    // Labels are validated but must not change the tensor output.
    let rows = [
        vec![0.0, 0.0, 10.0, 10.0, 0.1, 0.9],
        vec![50.0, 50.0, 60.0, 60.0, 0.8, 0.1],
    ];
    let plain = detect_from_output(&rows, &cfg(2)).unwrap();
    let mut labeled = cfg(2);
    labeled.labels = Some(names(2));
    labeled.output_format = OutputFormat::Tensor;
    let with_labels = detect_from_output(&rows, &labeled).unwrap();
    assert_eq!(with_labels.dtype, KAPSL_DTYPE_F32);
    assert_eq!(with_labels.shape, plain.shape);
    assert_eq!(with_labels.data, plain.data);

    // And mismatched labels are still rejected under `tensor`.
    labeled.labels = Some(names(3));
    assert!(labeled.validate().is_err());
}

#[test]
fn output_format_parses_and_unknown_value_is_rejected() {
    let config: DetectConfig =
        serde_yaml::from_str("num_classes: 1\nlabels: [a]\noutput_format: json").unwrap();
    assert_eq!(config.output_format, OutputFormat::Json);
    let config: DetectConfig =
        serde_yaml::from_str("num_classes: 1\noutput_format: tensor").unwrap();
    assert_eq!(config.output_format, OutputFormat::Tensor);
    for bad in ["xml", "JSON", "labeled", "\"\""] {
        let error = serde_yaml::from_str::<DetectConfig>(&format!(
            "num_classes: 1\nlabels: [a]\noutput_format: {bad}"
        ))
        .unwrap_err();
        assert!(
            error.to_string().contains("unknown variant"),
            "{bad}: {error}"
        );
    }
}

#[test]
fn json_output_format_requires_labels() {
    let mut config = cfg(2);
    config.output_format = OutputFormat::Json;
    let error = config.validate().unwrap_err();
    assert_eq!(error.0, KAPSL_STATUS_INVALID_ARGUMENT);
    assert!(error.1.contains("requires `labels`"), "{}", error.1);
}

// ---- manifest wiring, coordinate mapping, clipping ----

/// A detector package manifest, optionally with a 100x100 vision preprocessor.
fn manifest(detect: serde_json::Value, resize: Option<&str>) -> Manifest {
    let mut metadata = serde_json::json!({ "detect": detect });
    if let Some(resize) = resize {
        metadata["preprocess"] = serde_json::json!({
            "kind": "vision",
            "width": 100,
            "height": 100,
            "resize": resize
        });
    }
    serde_json::from_value(serde_json::json!({
        "project_name": "detector",
        "framework": "onnx",
        "version": "0.1.0",
        "created_at": "2026-09-01T00:00:00Z",
        "model_file": "detector.onnx",
        "format": "onnx",
        "model_type": "detector",
        "task": "detect",
        "metadata": metadata
    }))
    .unwrap()
}

/// The shared detector block: xyxy boxes, no objectness channel.
fn detector(num_classes: usize) -> serde_json::Value {
    serde_json::json!({
        "num_classes": num_classes,
        "box_format": "xyxy",
        "objectness": false
    })
}

/// The adapter's stateless request pipeline around a fixed raw model output:
/// the manifest's preprocessor, then its task postprocessor, composed the way
/// `lib.rs` composes them around the ORT run.
struct Pipeline {
    preprocessor: InputPreprocessor,
    task: TaskProcessor,
}

impl Pipeline {
    fn new(detect: serde_json::Value, resize: Option<&str>) -> FfiResult<Self> {
        let manifest = manifest(detect, resize);
        Ok(Self {
            task: TaskProcessor::from_manifest(&manifest)?,
            preprocessor: InputPreprocessor::from_manifest(&manifest)?,
        })
    }

    fn infer(&self, rows: &[Vec<f32>], input: BorrowedTensor<'_>) -> FfiResult<OwnedTensor> {
        let inputs = [input];
        let prepared = self.preprocessor.prepare(&inputs)?;
        let image = prepared
            .as_ref()
            .and_then(|prepared| prepared.image_placement());
        let effective = prepared
            .as_ref()
            .map_or_else(|| inputs.to_vec(), |prepared| prepared.views(&inputs));
        self.task
            .postprocess(raw_output(rows), &effective, image.as_ref())
    }

    /// One request carrying encoded image bytes.
    fn infer_image(&self, rows: &[Vec<f32>], encoded: &[u8]) -> FfiResult<OwnedTensor> {
        let shape = [encoded.len() as i64];
        self.infer(
            rows,
            BorrowedTensor {
                name: "input",
                dtype: KAPSL_DTYPE_U8,
                shape: &shape,
                data: encoded,
            },
        )
    }

    /// A request batch in which the model returns the same `rows` for every image.
    fn infer_image_batch(
        &self,
        rows: &[Vec<f32>],
        requests: &[Vec<u8>],
    ) -> FfiResult<Vec<OwnedTensor>> {
        let shapes = requests
            .iter()
            .map(|encoded| [encoded.len() as i64])
            .collect::<Vec<_>>();
        let inputs = requests
            .iter()
            .zip(&shapes)
            .map(|(encoded, shape)| {
                vec![BorrowedTensor {
                    name: "input",
                    dtype: KAPSL_DTYPE_U8,
                    shape,
                    data: encoded,
                }]
            })
            .collect::<Vec<_>>();
        let prepared = inputs
            .iter()
            .map(|inputs| self.preprocessor.prepare(inputs))
            .collect::<FfiResult<Vec<_>>>()?;
        let images = prepared
            .iter()
            .map(|prepared| {
                prepared
                    .as_ref()
                    .and_then(|prepared| prepared.image_placement())
            })
            .collect::<Vec<_>>();
        let effective = inputs
            .iter()
            .zip(&prepared)
            .map(|(inputs, prepared)| {
                prepared
                    .as_ref()
                    .map_or_else(|| inputs.clone(), |prepared| prepared.views(inputs))
            })
            .collect::<Vec<_>>();
        let outputs = requests.iter().map(|_| raw_output(rows)).collect();
        self.task.postprocess_batch(outputs, &effective, &images)
    }
}

fn png(width: u32, height: u32) -> Vec<u8> {
    let mut encoded = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image::RgbImage::new(width, height))
        .write_to(&mut encoded, image::ImageFormat::Png)
        .unwrap();
    encoded.into_inner()
}

#[test]
fn initialization_rejects_mismatched_labels_unknown_keys_and_unlabeled_json() {
    let rejected = |detect: serde_json::Value| {
        let error = Pipeline::new(detect, None).err().unwrap();
        assert_eq!(error.0, KAPSL_STATUS_INVALID_ARGUMENT, "{}", error.1);
        error.1
    };
    let message = rejected(serde_json::json!({"num_classes": 3, "labels": names(4)}));
    assert!(
        message.contains("labels has 4 entries but num_classes is 3"),
        "{message}"
    );
    let message = rejected(serde_json::json!({"num_classes": 2, "lables": ["a", "b"]}));
    assert!(message.contains("unknown field `lables`"), "{message}");
    let message = rejected(serde_json::json!({"num_classes": 2, "output_format": "json"}));
    assert!(message.contains("requires `labels`"), "{message}");
    let message = rejected(serde_json::json!({"num_classes": 1, "coordinates": "bogus"}));
    assert!(message.contains("unknown variant `bogus`"), "{message}");
    assert!(Pipeline::new(
        serde_json::json!({
            "num_classes": 2,
            "labels": names(2),
            "coordinates": "model",
            "output_format": "json"
        }),
        None
    )
    .is_ok());
}

#[test]
fn letterbox_boxes_map_back_to_original_image_pixels() {
    // 200x100 source into a 100x100 box: scale 0.5, fit 100x50, pad_y 25.
    // Model-space box (10,35)-(60,60) -> source (20,20)-(120,70).
    let pipeline = Pipeline::new(detector(1), Some("letterbox")).unwrap();
    let output = pipeline
        .infer_image(&[vec![10.0, 35.0, 60.0, 60.0, 0.9]], &png(200, 100))
        .unwrap();
    assert_eq!(
        decode_rows(&output),
        vec![[20.0, 20.0, 120.0, 70.0, 0.9, 0.0]]
    );
}

#[test]
fn stretch_boxes_map_back_with_independent_axis_scales() {
    // 200x50 stretched to 100x100: x scale 0.5, y scale 2.
    let pipeline = Pipeline::new(detector(1), Some("stretch")).unwrap();
    let output = pipeline
        .infer_image(&[vec![10.0, 20.0, 50.0, 80.0, 0.9]], &png(200, 50))
        .unwrap();
    assert_eq!(
        decode_rows(&output),
        vec![[20.0, 10.0, 100.0, 40.0, 0.9, 0.0]]
    );
}

#[test]
fn boxes_are_clipped_to_the_original_image() {
    // 200x100 source, letterbox. This box overhangs every edge in model space.
    let pipeline = Pipeline::new(detector(1), Some("letterbox")).unwrap();
    let output = pipeline
        .infer_image(&[vec![-20.0, -10.0, 130.0, 120.0, 0.9]], &png(200, 100))
        .unwrap();
    assert_eq!(
        decode_rows(&output),
        vec![[0.0, 0.0, 200.0, 100.0, 0.9, 0.0]]
    );
}

#[test]
fn boxes_entirely_in_the_letterbox_padding_are_dropped() {
    // Padding occupies model y in [0,25) and [75,100) for a 200x100 source.
    let pipeline = Pipeline::new(detector(1), Some("letterbox")).unwrap();
    let output = pipeline
        .infer_image(
            &[
                vec![10.0, 2.0, 60.0, 20.0, 0.9],  // fully in the top pad
                vec![10.0, 30.0, 60.0, 60.0, 0.8], // real content
            ],
            &png(200, 100),
        )
        .unwrap();
    let rows = decode_rows(&output);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][4], 0.8);
}

#[test]
fn model_coordinates_flag_keeps_model_space_and_clips_to_the_input() {
    let mut detect = detector(1);
    detect["coordinates"] = "model".into();
    let pipeline = Pipeline::new(detect, Some("letterbox")).unwrap();
    let output = pipeline
        .infer_image(&[vec![-5.0, 35.0, 60.0, 130.0, 0.9]], &png(200, 100))
        .unwrap();
    assert_eq!(
        decode_rows(&output),
        vec![[0.0, 35.0, 60.0, 100.0, 0.9, 0.0]]
    );
}

#[test]
fn without_image_input_boxes_stay_in_model_space_unclipped() {
    let pipeline = Pipeline::new(detector(1), None).unwrap();
    let shape = [1_i64, 1];
    let data = 0.0_f32.to_ne_bytes();
    let output = pipeline
        .infer(
            &[vec![-5.0, 0.0, 600.0, 10.0, 0.9]],
            BorrowedTensor {
                name: "input",
                dtype: KAPSL_DTYPE_F32,
                shape: &shape,
                data: &data,
            },
        )
        .unwrap();
    assert_eq!(
        decode_rows(&output),
        vec![[-5.0, 0.0, 600.0, 10.0, 0.9, 0.0]]
    );
}

#[test]
fn image_input_rejects_a_non_image_request() {
    let pipeline = Pipeline::new(detector(1), Some("letterbox")).unwrap();
    let error = pipeline
        .infer_image(&[vec![0.0, 0.0, 10.0, 10.0, 0.9]], &[1, 2, 3, 4])
        .err()
        .unwrap();
    assert_eq!(error.0, KAPSL_STATUS_INVALID_ARGUMENT, "{}", error.1);
}

#[test]
fn batch_maps_each_request_with_its_own_image_size() {
    let pipeline = Pipeline::new(detector(1), Some("letterbox")).unwrap();
    let outputs = pipeline
        .infer_image_batch(
            &[vec![10.0, 35.0, 60.0, 60.0, 0.9]],
            &[png(200, 100), png(100, 200)],
        )
        .unwrap();
    // 200x100: scale 0.5, pad_y 25 -> (20,20,120,70).
    assert_eq!(
        decode_rows(&outputs[0]),
        vec![[20.0, 20.0, 120.0, 70.0, 0.9, 0.0]]
    );
    // 100x200: scale 0.5, pad_x 25 -> x=(10-25)/0.5<0 clips to 0, x2=(60-25)/0.5=70;
    // y=(35-0)/0.5=70, y2=120.
    assert_eq!(
        decode_rows(&outputs[1]),
        vec![[0.0, 70.0, 70.0, 120.0, 0.9, 0.0]]
    );
}

#[test]
fn batch_postprocessing_requires_one_image_placement_slot_per_request() {
    let pipeline = Pipeline::new(detector(1), None).unwrap();
    let rows = [vec![0.0, 0.0, 10.0, 10.0, 0.9]];
    let error = pipeline
        .task
        .postprocess_batch(vec![raw_output(&rows)], &[Vec::new()], &[])
        .err()
        .unwrap();
    assert!(
        error.1.contains("batch result length mismatch"),
        "{}",
        error.1
    );
}

#[test]
fn json_format_works_with_original_coordinates_end_to_end() {
    let mut detect = detector(1);
    detect["labels"] = serde_json::json!(["thing"]);
    detect["output_format"] = "json".into();
    let pipeline = Pipeline::new(detect, Some("letterbox")).unwrap();
    let output = pipeline
        .infer_image(&[vec![10.0, 35.0, 60.0, 60.0, 0.9]], &png(200, 100))
        .unwrap();
    assert_eq!(output.dtype, KAPSL_DTYPE_UTF8);
    let value: serde_json::Value = serde_json::from_slice(&output.data).unwrap();
    assert_eq!(
        value[0]["box"],
        serde_json::json!([20.0, 20.0, 120.0, 70.0])
    );
    assert_eq!(value[0]["label"], "thing");
    // The wire bytes are the embedded backend's serde_json encoding of the
    // same row: field order, float formatting and no whitespace.
    assert_eq!(
        std::str::from_utf8(&output.data).unwrap(),
        r#"[{"box":[20.0,20.0,120.0,70.0],"score":0.9,"class_id":0,"label":"thing"}]"#
    );
}

#[test]
fn model_info_reports_the_selected_detection_output_format() {
    let adjusted = |detect: serde_json::Value| {
        let task = Pipeline::new(detect, None).unwrap().task;
        let mut info = EngineModelInfo {
            input_names: vec!["images".to_string()],
            output_names: vec!["output0".to_string()],
            input_shapes: vec![vec![1, 3, 100, 100]],
            output_shapes: vec![vec![1, 25200, 6]],
            input_dtypes: vec!["float32".to_string()],
            output_dtypes: vec!["float32".to_string()],
            framework: Some("onnx".to_string()),
            model_version: None,
            peak_concurrency: Some(1),
        };
        task.adjust_model_info(&mut info);
        (info.output_names, info.output_shapes, info.output_dtypes)
    };
    assert_eq!(
        adjusted(detector(1)),
        (
            vec!["detections".to_string()],
            vec![vec![-1, 6]],
            vec!["float32".to_string()]
        )
    );
    let mut json = detector(1);
    json["labels"] = serde_json::json!(["thing"]);
    json["output_format"] = "json".into();
    assert_eq!(
        adjusted(json),
        (
            vec!["detections".to_string()],
            vec![vec![-1]],
            vec!["string".to_string()]
        )
    );
}
