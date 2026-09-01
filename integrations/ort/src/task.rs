//! Manifest-selected ONNX task postprocessing.
//!
//! The generic engine host deliberately does not know ORT task semantics. This
//! module keeps embedding pooling, classification softmax, YOLO decoding/NMS,
//! and greedy CTC decoding on the integration side of the ABI.

use crate::tensor::{BorrowedTensor, OwnedTensor};
use crate::{backend_error, invalid_argument, FfiResult};
use kapsl_backend_abi::{KAPSL_DTYPE_F32, KAPSL_DTYPE_I32, KAPSL_DTYPE_I64};
use kapsl_core::loader::Manifest;
use kapsl_core::EngineKind;
use kapsl_engine_api::EngineModelInfo;
use serde::Deserialize;

#[derive(Debug)]
pub(crate) enum TaskProcessor {
    Forward,
    Embed { normalize: bool },
    Classify { apply_softmax: bool },
    Detect(DetectConfig),
    Transcribe(TranscribeConfig),
}

impl TaskProcessor {
    pub(crate) fn from_manifest(manifest: &Manifest) -> FfiResult<Self> {
        EngineKind::validate(manifest)
            .map_err(|error| invalid_argument(format!("invalid model contract: {error}")))?;
        match EngineKind::resolve(manifest) {
            EngineKind::OnnxForward => Ok(Self::Forward),
            EngineKind::OnnxEmbed => Ok(Self::Embed {
                normalize: metadata_bool(manifest, "embed", "normalize", true),
            }),
            EngineKind::OnnxClassify => Ok(Self::Classify {
                apply_softmax: metadata_bool(manifest, "classify", "apply_softmax", true),
            }),
            EngineKind::OnnxDetect => {
                let value = metadata_section(manifest, "detect").ok_or_else(|| {
                    invalid_argument(
                        "ONNX detect task requires a metadata.detect configuration block",
                    )
                })?;
                let config: DetectConfig =
                    serde_yaml::from_value(value.clone()).map_err(|error| {
                        invalid_argument(format!("decode metadata.detect configuration: {error}"))
                    })?;
                config.validate()?;
                Ok(Self::Detect(config))
            }
            EngineKind::OnnxTranscribe => {
                let config = metadata_section(manifest, "transcribe")
                    .map(|value| serde_yaml::from_value(value.clone()))
                    .transpose()
                    .map_err(|error| {
                        invalid_argument(format!(
                            "decode metadata.transcribe configuration: {error}"
                        ))
                    })?
                    .unwrap_or_default();
                Ok(Self::Transcribe(config))
            }
            EngineKind::OnnxGenerate => Err((
                kapsl_backend_abi::KAPSL_STATUS_UNSUPPORTED,
                "ONNX generation requires the separately certified generation profile".to_string(),
            )),
            other => Err(invalid_argument(format!(
                "ORT adapter cannot execute model kind `{}`",
                other.label()
            ))),
        }
    }

    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Embed { .. } => "embed",
            Self::Classify { .. } => "classify",
            Self::Detect(_) => "detect",
            Self::Transcribe(_) => "transcribe",
        }
    }

    pub(crate) fn postprocess(
        &self,
        output: OwnedTensor,
        inputs: &[BorrowedTensor<'_>],
    ) -> FfiResult<OwnedTensor> {
        match self {
            Self::Forward => Ok(output),
            Self::Embed { normalize } => embed(output, inputs, *normalize),
            Self::Classify { apply_softmax } => classify(output, *apply_softmax),
            Self::Detect(config) => detect(output, config),
            Self::Transcribe(config) => transcribe(output, config),
        }
    }

    pub(crate) fn postprocess_batch(
        &self,
        outputs: Vec<OwnedTensor>,
        inputs: &[Vec<BorrowedTensor<'_>>],
    ) -> FfiResult<Vec<OwnedTensor>> {
        if outputs.len() != inputs.len() {
            return Err(backend_error(format!(
                "{} batch result length mismatch: expected {}, got {}",
                self.label(),
                inputs.len(),
                outputs.len()
            )));
        }
        outputs
            .into_iter()
            .zip(inputs)
            .map(|(output, request_inputs)| self.postprocess(output, request_inputs))
            .collect()
    }

    pub(crate) fn adjust_model_info(&self, info: &mut EngineModelInfo) {
        match self {
            Self::Forward => {}
            Self::Embed { .. } => {
                info.output_names = vec!["embedding".to_string()];
                if let Some(shape) = info.output_shapes.first() {
                    info.output_shapes = vec![match shape.as_slice() {
                        [batch, dim] => vec![*batch, *dim],
                        [batch, _, dim] => vec![*batch, *dim],
                        _ => vec![-1, -1],
                    }];
                }
                info.output_dtypes = vec!["float32".to_string()];
            }
            Self::Classify { .. } => {
                info.output_names = vec!["probabilities".to_string()];
                if let Some(shape) = info.output_shapes.first() {
                    if let [classes] = shape.as_slice() {
                        info.output_shapes = vec![vec![1, *classes]];
                    }
                }
                info.output_dtypes = vec!["float32".to_string()];
            }
            Self::Detect(_) => {
                info.output_names = vec!["detections".to_string()];
                info.output_shapes = vec![vec![-1, 6]];
                info.output_dtypes = vec!["float32".to_string()];
            }
            Self::Transcribe(_) => {
                info.output_names = vec!["tokens".to_string()];
                info.output_shapes = vec![vec![-1]];
                info.output_dtypes = vec!["int32".to_string()];
            }
        }
    }
}

fn metadata_section<'a>(manifest: &'a Manifest, section: &str) -> Option<&'a serde_yaml::Value> {
    manifest.metadata.as_ref()?.get(section)
}

fn metadata_bool(manifest: &Manifest, section: &str, key: &str, default: bool) -> bool {
    metadata_section(manifest, section)
        .and_then(|value| value.get(key))
        .and_then(serde_yaml::Value::as_bool)
        .unwrap_or(default)
}

fn embed(
    output: OwnedTensor,
    inputs: &[BorrowedTensor<'_>],
    normalize: bool,
) -> FfiResult<OwnedTensor> {
    let values = f32_values(&output, "embedding")?;
    match output.shape.as_slice() {
        [batch, dim] => {
            let batch = dimension(*batch, "embedding batch")?;
            let dim = dimension(*dim, "embedding width")?;
            require_nonzero(batch, "embedding batch")?;
            require_nonzero(dim, "embedding width")?;
            require_elements(values.len(), &[batch, dim], "embedding")?;
            let mut pooled = values;
            if normalize {
                l2_normalize_rows(&mut pooled, batch, dim);
            }
            Ok(f32_tensor(
                "embedding",
                vec![batch as i64, dim as i64],
                pooled,
            ))
        }
        [batch, sequence, dim] => {
            let batch = dimension(*batch, "embedding batch")?;
            let sequence = dimension(*sequence, "embedding sequence")?;
            let dim = dimension(*dim, "embedding width")?;
            require_nonzero(batch, "embedding batch")?;
            require_nonzero(sequence, "embedding sequence")?;
            require_nonzero(dim, "embedding width")?;
            require_elements(values.len(), &[batch, sequence, dim], "embedding")?;
            let mask = attention_mask(inputs, batch, sequence);
            let mut pooled = masked_mean_pool(&values, batch, sequence, dim, &mask);
            if normalize {
                l2_normalize_rows(&mut pooled, batch, dim);
            }
            Ok(f32_tensor(
                "embedding",
                vec![batch as i64, dim as i64],
                pooled,
            ))
        }
        shape => Err(backend_error(format!(
            "embedding expects [batch, dim] or [batch, sequence, dim], got {shape:?}"
        ))),
    }
}

fn masked_mean_pool(
    hidden: &[f32],
    batch: usize,
    sequence: usize,
    dim: usize,
    mask: &[f32],
) -> Vec<f32> {
    let mut pooled = vec![0.0; batch.saturating_mul(dim)];
    for batch_index in 0..batch {
        let mut denominator = 0.0_f32;
        for sequence_index in 0..sequence {
            let weight = mask
                .get(batch_index * sequence + sequence_index)
                .copied()
                .unwrap_or(1.0);
            if weight == 0.0 {
                continue;
            }
            denominator += weight;
            let hidden_offset = (batch_index * sequence + sequence_index) * dim;
            let output_offset = batch_index * dim;
            for column in 0..dim {
                pooled[output_offset + column] += weight * hidden[hidden_offset + column];
            }
        }
        let denominator = denominator.max(1e-9);
        let output_offset = batch_index * dim;
        for column in 0..dim {
            pooled[output_offset + column] /= denominator;
        }
    }
    pooled
}

fn l2_normalize_rows(values: &mut [f32], rows: usize, columns: usize) {
    for row in values.chunks_exact_mut(columns).take(rows) {
        let norm = row
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
            .max(1e-12);
        for value in row {
            *value /= norm;
        }
    }
}

fn attention_mask(inputs: &[BorrowedTensor<'_>], batch: usize, sequence: usize) -> Vec<f32> {
    let expected = batch.saturating_mul(sequence);
    let Some(mask) = inputs
        .iter()
        .skip(1)
        .find(|input| input.name.contains("attention_mask"))
    else {
        return vec![1.0; expected];
    };
    let values = match mask.dtype {
        KAPSL_DTYPE_F32 => parse_f32(mask.data),
        KAPSL_DTYPE_I64 => mask
            .data
            .chunks_exact(8)
            .map(|bytes| i64::from_ne_bytes(bytes.try_into().expect("eight bytes")) as f32)
            .collect(),
        KAPSL_DTYPE_I32 => mask
            .data
            .chunks_exact(4)
            .map(|bytes| i32::from_ne_bytes(bytes.try_into().expect("four bytes")) as f32)
            .collect(),
        _ => Vec::new(),
    };
    if values.len() == expected {
        values
    } else {
        vec![1.0; expected]
    }
}

fn classify(output: OwnedTensor, apply_softmax: bool) -> FfiResult<OwnedTensor> {
    let mut values = f32_values(&output, "classification")?;
    let (batch, classes, shape) = match output.shape.as_slice() {
        [classes] => {
            let classes = dimension(*classes, "classification classes")?;
            (1, classes, vec![1, classes as i64])
        }
        [batch, classes] => {
            let batch = dimension(*batch, "classification batch")?;
            let classes = dimension(*classes, "classification classes")?;
            (batch, classes, output.shape.clone())
        }
        shape => {
            return Err(backend_error(format!(
                "classification expects [classes] or [batch, classes], got {shape:?}"
            )))
        }
    };
    if classes == 0 {
        return Err(backend_error("classification output has zero classes"));
    }
    require_elements(values.len(), &[batch, classes], "classification")?;
    if apply_softmax {
        softmax_rows(&mut values, classes);
    }
    Ok(f32_tensor("probabilities", shape, values))
}

fn softmax_rows(values: &mut [f32], classes: usize) {
    for row in values.chunks_exact_mut(classes) {
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for value in row.iter_mut() {
            *value = (*value - maximum).exp();
            sum += *value;
        }
        let sum = sum.max(1e-12);
        for value in row {
            *value /= sum;
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum BoxFormat {
    #[default]
    Xywh,
    Xyxy,
}

fn default_score_threshold() -> f32 {
    0.25
}

fn default_iou_threshold() -> f32 {
    0.45
}

fn default_max_detections() -> usize {
    300
}

fn default_objectness() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub(crate) struct DetectConfig {
    num_classes: usize,
    #[serde(default = "default_score_threshold")]
    score_threshold: f32,
    #[serde(default = "default_iou_threshold")]
    iou_threshold: f32,
    #[serde(default = "default_max_detections")]
    max_detections: usize,
    #[serde(default)]
    box_format: BoxFormat,
    #[serde(default = "default_objectness")]
    objectness: bool,
    #[serde(default)]
    transposed: bool,
    #[serde(default)]
    class_agnostic: bool,
}

impl DetectConfig {
    fn validate(&self) -> FfiResult<()> {
        if self.num_classes == 0 {
            return Err(invalid_argument("detect num_classes must be non-zero"));
        }
        if !self.score_threshold.is_finite()
            || !self.iou_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.iou_threshold)
        {
            return Err(invalid_argument(
                "detect thresholds must be finite and iou_threshold must be within 0..=1",
            ));
        }
        if self.max_detections == 0 {
            return Err(invalid_argument("detect max_detections must be non-zero"));
        }
        Ok(())
    }

    fn channels(&self) -> usize {
        4 + usize::from(self.objectness) + self.num_classes
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    score: f32,
    class: usize,
}

fn detect(output: OwnedTensor, config: &DetectConfig) -> FfiResult<OwnedTensor> {
    let values = f32_values(&output, "detection")?;
    let (first, second) = match output.shape.as_slice() {
        [first, second] => (
            dimension(*first, "detection grid")?,
            dimension(*second, "detection grid")?,
        ),
        [1, first, second] => (
            dimension(*first, "detection grid")?,
            dimension(*second, "detection grid")?,
        ),
        shape => {
            return Err(backend_error(format!(
                "detection expects [anchors, channels] or [1, anchors, channels], got {shape:?}"
            )))
        }
    };
    let (anchors, channels) = if config.transposed {
        (second, first)
    } else {
        (first, second)
    };
    if channels != config.channels() {
        return Err(backend_error(format!(
            "detection output has {channels} channels, configuration requires {}",
            config.channels()
        )));
    }
    require_elements(values.len(), &[anchors, channels], "detection")?;
    let at = |anchor: usize, channel: usize| {
        values[if config.transposed {
            channel * anchors + anchor
        } else {
            anchor * channels + channel
        }]
    };
    let class_offset = 4 + usize::from(config.objectness);
    let mut candidates = Vec::new();
    for anchor in 0..anchors {
        let mut best_class = 0;
        let mut best_score = f32::NEG_INFINITY;
        for class in 0..config.num_classes {
            let score = at(anchor, class_offset + class);
            if score > best_score {
                best_score = score;
                best_class = class;
            }
        }
        let score = if config.objectness {
            at(anchor, 4) * best_score
        } else {
            best_score
        };
        if score < config.score_threshold {
            continue;
        }
        let (a, b, c, d) = (at(anchor, 0), at(anchor, 1), at(anchor, 2), at(anchor, 3));
        let (x1, y1, x2, y2) = match config.box_format {
            BoxFormat::Xywh => (a - c / 2.0, b - d / 2.0, a + c / 2.0, b + d / 2.0),
            BoxFormat::Xyxy => (a, b, c, d),
        };
        candidates.push(Candidate {
            x1,
            y1,
            x2,
            y2,
            score,
            class: best_class,
        });
    }
    let mut kept = non_max_suppression(candidates, config.iou_threshold, config.class_agnostic);
    kept.truncate(config.max_detections);
    let mut values = Vec::with_capacity(kept.len() * 6);
    for detection in kept {
        values.extend_from_slice(&[
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
            detection.score,
            detection.class as f32,
        ]);
    }
    Ok(f32_tensor(
        "detections",
        vec![(values.len() / 6) as i64, 6],
        values,
    ))
}

fn non_max_suppression(
    mut candidates: Vec<Candidate>,
    iou_threshold: f32,
    class_agnostic: bool,
) -> Vec<Candidate> {
    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept: Vec<Candidate> = Vec::new();
    'candidate: for candidate in candidates {
        for existing in &kept {
            if (class_agnostic || existing.class == candidate.class)
                && intersection_over_union(existing, &candidate) > iou_threshold
            {
                continue 'candidate;
            }
        }
        kept.push(candidate);
    }
    kept
}

fn intersection_over_union(left: &Candidate, right: &Candidate) -> f32 {
    let width = (left.x2.min(right.x2) - left.x1.max(right.x1)).max(0.0);
    let height = (left.y2.min(right.y2) - left.y1.max(right.y1)).max(0.0);
    let intersection = width * height;
    let left_area = (left.x2 - left.x1).max(0.0) * (left.y2 - left.y1).max(0.0);
    let right_area = (right.x2 - right.x1).max(0.0) * (right.y2 - right.y1).max(0.0);
    let union = left_area + right_area - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn default_collapse_repeats() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub(crate) struct TranscribeConfig {
    #[serde(default)]
    blank_id: usize,
    #[serde(default = "default_collapse_repeats")]
    collapse_repeats: bool,
}

impl Default for TranscribeConfig {
    fn default() -> Self {
        Self {
            blank_id: 0,
            collapse_repeats: true,
        }
    }
}

fn transcribe(output: OwnedTensor, config: &TranscribeConfig) -> FfiResult<OwnedTensor> {
    let values = f32_values(&output, "transcription")?;
    let (time, vocabulary) = match output.shape.as_slice() {
        [time, vocabulary] => (
            dimension(*time, "transcription time")?,
            dimension(*vocabulary, "transcription vocabulary")?,
        ),
        [1, time, vocabulary] => (
            dimension(*time, "transcription time")?,
            dimension(*vocabulary, "transcription vocabulary")?,
        ),
        shape => {
            return Err(backend_error(format!(
                "transcription expects [time, vocabulary] or [1, time, vocabulary], got {shape:?}"
            )))
        }
    };
    if vocabulary == 0 || config.blank_id >= vocabulary {
        return Err(backend_error(format!(
            "transcription blank_id {} is invalid for vocabulary size {vocabulary}",
            config.blank_id
        )));
    }
    require_elements(values.len(), &[time, vocabulary], "transcription")?;
    let mut tokens = Vec::new();
    let mut previous = None;
    for row in values.chunks_exact(vocabulary).take(time) {
        let mut best = 0;
        for index in 1..row.len() {
            if row[index] > row[best] {
                best = index;
            }
        }
        if config.collapse_repeats && previous == Some(best) {
            continue;
        }
        previous = Some(best);
        if best != config.blank_id {
            tokens.push(best as i32);
        }
    }
    Ok(i32_tensor("tokens", vec![tokens.len() as i64], tokens))
}

fn f32_values(output: &OwnedTensor, task: &str) -> FfiResult<Vec<f32>> {
    if output.dtype != KAPSL_DTYPE_F32 || !output.data.len().is_multiple_of(4) {
        return Err(backend_error(format!(
            "{task} requires a float32 ONNX output"
        )));
    }
    Ok(parse_f32(&output.data))
}

fn parse_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("four bytes")))
        .collect()
}

fn dimension(value: i64, label: &str) -> FfiResult<usize> {
    usize::try_from(value).map_err(|_| backend_error(format!("{label} dimension is {value}")))
}

fn require_nonzero(value: usize, label: &str) -> FfiResult<()> {
    if value == 0 {
        return Err(backend_error(format!("{label} dimension is zero")));
    }
    Ok(())
}

fn require_elements(actual: usize, dimensions: &[usize], label: &str) -> FfiResult<()> {
    let expected = dimensions.iter().try_fold(1_usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or_else(|| backend_error(format!("{label} shape overflows")))
    })?;
    if actual != expected {
        return Err(backend_error(format!(
            "{label} output has {actual} values, shape implies {expected}"
        )));
    }
    Ok(())
}

fn f32_tensor(name: &str, shape: Vec<i64>, values: Vec<f32>) -> OwnedTensor {
    let mut data = Vec::with_capacity(values.len() * 4);
    for value in values {
        data.extend_from_slice(&value.to_ne_bytes());
    }
    OwnedTensor {
        name: name.to_string(),
        dtype: KAPSL_DTYPE_F32,
        shape,
        data,
    }
}

fn i32_tensor(name: &str, shape: Vec<i64>, values: Vec<i32>) -> OwnedTensor {
    let mut data = Vec::with_capacity(values.len() * 4);
    for value in values {
        data.extend_from_slice(&value.to_ne_bytes());
    }
    OwnedTensor {
        name: name.to_string(),
        dtype: KAPSL_DTYPE_I32,
        shape,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_output(shape: Vec<i64>, values: &[f32]) -> OwnedTensor {
        f32_tensor("raw", shape, values.to_vec())
    }

    fn values(output: &OwnedTensor) -> Vec<f32> {
        parse_f32(&output.data)
    }

    #[test]
    fn classification_softmax_is_stable() {
        let output = classify(f32_output(vec![1, 3], &[1000.0, 1001.0, 999.0]), true).unwrap();
        let probabilities = values(&output);
        assert!(probabilities.iter().all(|value| value.is_finite()));
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(probabilities[1] > probabilities[0]);
    }

    #[test]
    fn embedding_uses_attention_mask_and_normalizes() {
        let mask_data = [1_i64, 0]
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        let mask_shape = [1_i64, 2];
        let mask = BorrowedTensor {
            name: "attention_mask",
            dtype: KAPSL_DTYPE_I64,
            shape: &mask_shape,
            data: &mask_data,
        };
        let output = embed(
            f32_output(vec![1, 2, 2], &[3.0, 4.0, 30.0, 40.0]),
            &[
                BorrowedTensor {
                    name: "input",
                    dtype: KAPSL_DTYPE_I64,
                    shape: &mask_shape,
                    data: &mask_data,
                },
                mask,
            ],
            true,
        )
        .unwrap();
        assert_eq!(output.shape, vec![1, 2]);
        assert_eq!(values(&output), vec![0.6, 0.8]);
    }

    #[test]
    fn detection_applies_per_class_nms() {
        let config = DetectConfig {
            num_classes: 2,
            score_threshold: 0.25,
            iou_threshold: 0.45,
            max_detections: 300,
            box_format: BoxFormat::Xyxy,
            objectness: false,
            transposed: false,
            class_agnostic: false,
        };
        let output = detect(
            f32_output(
                vec![3, 6],
                &[
                    0.0, 0.0, 10.0, 10.0, 0.9, 0.1, 0.5, 0.5, 10.5, 10.5, 0.8, 0.1, 0.5, 0.5, 10.5,
                    10.5, 0.1, 0.7,
                ],
            ),
            &config,
        )
        .unwrap();
        assert_eq!(output.shape, vec![2, 6]);
        let rows = values(&output);
        assert_eq!(rows[4], 0.9);
        assert_eq!(rows[11], 1.0);
    }

    #[test]
    fn transcription_collapses_repeats_and_blanks() {
        let config = TranscribeConfig::default();
        let output = transcribe(
            f32_output(
                vec![5, 3],
                &[
                    0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                ],
            ),
            &config,
        )
        .unwrap();
        let tokens = output
            .data
            .chunks_exact(4)
            .map(|bytes| i32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(tokens, vec![1, 2]);
    }
}
