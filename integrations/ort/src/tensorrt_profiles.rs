//! Versioned, model-local TensorRT generation shape profiles.
//!
//! The pinned ORT parser represents multiple profiles by repeating each input
//! name in one comma-separated list. Never configure these through process-wide
//! environment variables: different replicas may load different model shapes.

use serde::Deserialize;
use std::collections::BTreeMap;

const MAX_MODELS: usize = 256;
const MAX_PROFILES: usize = 16;
const MAX_INPUTS: usize = 1024;
const MAX_RANK: usize = 8;

type Shapes = BTreeMap<String, Vec<i64>>;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShapeProfile {
    min: Shapes,
    opt: Shapes,
    max: Shapes,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GenerationProfiles {
    version: u32,
    models: BTreeMap<String, Vec<ShapeProfile>>,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct ProviderProfileStrings {
    pub(crate) min: String,
    pub(crate) opt: String,
    pub(crate) max: String,
}

impl GenerationProfiles {
    pub(crate) fn from_metadata(metadata: Option<&serde_yaml::Value>) -> Result<Self, String> {
        let value = metadata
            .and_then(|value| value.get("ort"))
            .and_then(|value| value.get("tensorrt"))
            .ok_or_else(|| {
                "TensorRT generation requires explicit per-model shape profiles in metadata.ort.tensorrt"
                    .to_string()
            })?;
        let profiles: Self = serde_yaml::from_value(value.clone())
            .map_err(|error| format!("invalid metadata.ort.tensorrt: {error}"))?;
        profiles.validate()?;
        Ok(profiles)
    }

    fn validate(&self) -> Result<(), String> {
        if self.version != 1 {
            return Err(format!(
                "unsupported TensorRT profile version {}",
                self.version
            ));
        }
        if self.models.is_empty() || self.models.len() > MAX_MODELS {
            return Err(format!(
                "TensorRT profiles require 1..={MAX_MODELS} model files"
            ));
        }
        for (path, profiles) in &self.models {
            validate_model_key(path)?;
            if profiles.is_empty() || profiles.len() > MAX_PROFILES {
                return Err(format!(
                    "TensorRT model {path} requires 1..={MAX_PROFILES} profiles"
                ));
            }
            for (index, profile) in profiles.iter().enumerate() {
                profile
                    .validate()
                    .map_err(|error| format!("TensorRT model {path} profile {index}: {error}"))?;
                if !profile.min.keys().eq(profiles[0].min.keys()) {
                    return Err(format!(
                        "TensorRT model {path} profiles name different inputs"
                    ));
                }
                if profile
                    .min
                    .iter()
                    .any(|(name, shape)| shape.len() != profiles[0].min[name].len())
                {
                    return Err(format!(
                        "TensorRT model {path} profiles disagree on input ranks"
                    ));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn for_model(&self, relative_path: &str) -> Result<ProviderProfileStrings, String> {
        validate_model_key(relative_path)?;
        let profiles = self.models.get(relative_path).ok_or_else(|| {
            format!("no explicit TensorRT generation profiles for model file {relative_path}")
        })?;
        let encode = |select: fn(&ShapeProfile) -> &Shapes| {
            profiles
                .iter()
                .flat_map(|profile| select(profile).iter())
                .map(|(name, shape)| {
                    let dimensions = shape
                        .iter()
                        .map(i64::to_string)
                        .collect::<Vec<_>>()
                        .join("x");
                    format!("{name}:{dimensions}")
                })
                .collect::<Vec<_>>()
                .join(",")
        };
        Ok(ProviderProfileStrings {
            min: encode(|profile| &profile.min),
            opt: encode(|profile| &profile.opt),
            max: encode(|profile| &profile.max),
        })
    }
}

fn validate_model_key(path: &str) -> Result<(), String> {
    if path.is_empty()
        || path.len() > 4096
        || path.contains(['\\', ':'])
        || path.chars().any(char::is_control)
        || path.split('/').any(|part| matches!(part, "" | "." | ".."))
    {
        return Err(
            "TensorRT profile model keys must be normalized relative paths using '/'".into(),
        );
    }
    Ok(())
}

impl ShapeProfile {
    fn validate(&self) -> Result<(), String> {
        if self.min.is_empty() || self.min.len() > MAX_INPUTS {
            return Err(format!(
                "each profile requires 1..={MAX_INPUTS} named inputs"
            ));
        }
        if !self.min.keys().eq(self.opt.keys()) || !self.min.keys().eq(self.max.keys()) {
            return Err("min, opt and max must name the same inputs".into());
        }
        for (name, min) in &self.min {
            if name.is_empty()
                || name.len() > 1024
                || name.contains([':', ',', ';'])
                || name
                    .chars()
                    .any(|character| character.is_whitespace() || character.is_control())
            {
                return Err("input names must not contain profile delimiters, whitespace or control characters".into());
            }
            let opt = &self.opt[name];
            let max = &self.max[name];
            if min.is_empty()
                || min.len() > MAX_RANK
                || min.len() != opt.len()
                || min.len() != max.len()
            {
                return Err(format!(
                    "input {name} needs matching ranks in 1..={MAX_RANK}"
                ));
            }
            for ((min, opt), max) in min.iter().zip(opt).zip(max) {
                // ORT 1.23.2 parses dimensions with std::stoi, including zero
                // dimensions for an initially empty generation KV cache.
                if *min < 0 || min > opt || opt > max || *max > i64::from(i32::MAX) {
                    return Err(format!(
                        "input {name} requires 0 <= min <= opt <= max <= i32::MAX"
                    ));
                }
            }
        }
        for (label, shapes) in [("min", &self.min), ("opt", &self.opt), ("max", &self.max)] {
            validate_standard_decoder_shapes(shapes)
                .map_err(|error| format!("{label} shapes: {error}"))?;
        }
        Ok(())
    }
}

fn validate_standard_decoder_shapes(shapes: &Shapes) -> Result<(), String> {
    // Validate the common decoder contract that caused the Vast failure.
    // Other graph-specific relationships remain ORT/TensorRT load validation.
    let Some(ids) = shapes.get("input_ids").filter(|shape| shape.len() == 2) else {
        return Ok(());
    };
    if ids[0] == 0 || ids[1] == 0 {
        return Err("input_ids requires a nonempty batch and sequence".into());
    }
    if let Some(positions) = shapes.get("position_ids") {
        if positions != ids {
            return Err("position_ids must match input_ids".into());
        }
    }
    let mut past_length = None;
    for (name, shape) in shapes {
        let parts = name.split('.').collect::<Vec<_>>();
        if parts.len() != 3
            || parts[0] != "past_key_values"
            || parts[1].parse::<u32>().is_err()
            || !matches!(parts[2], "key" | "value")
        {
            continue;
        }
        if shape.len() != 4 || shape[0] != ids[0] || shape[1] == 0 || shape[3] == 0 {
            return Err(format!(
                "{name} must have [batch, heads, past, width] shape matching input_ids"
            ));
        }
        if past_length.is_some_and(|past| past != shape[2]) {
            return Err("past_key_values inputs must have the same past length".into());
        }
        past_length = Some(shape[2]);
    }
    if let (Some(past), Some(mask)) = (past_length, shapes.get("attention_mask")) {
        if mask.len() != 2 || mask[0] != ids[0] || mask[1] != ids[1] + past {
            return Err("attention_mask must have [batch, sequence + past] shape".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn shapes(sequence: i64, past: i64) -> Value {
        json!({
            "input_ids": [1, sequence], "position_ids": [1, sequence],
            "attention_mask": [1, sequence + past],
            "past_key_values.0.key": [1, 2, past, 64],
            "past_key_values.0.value": [1, 2, past, 64]
        })
    }

    fn metadata(profile: Value) -> serde_yaml::Value {
        serde_yaml::to_value(json!({"ort": {"tensorrt": {"version": 1, "models": {
            "onnx/model.onnx": [profile]
        }}}}))
        .unwrap()
    }

    fn valid_profile() -> Value {
        json!({"min": shapes(1, 0), "opt": shapes(9, 0), "max": shapes(16, 16)})
    }

    #[test]
    fn documented_tiny_gpt2_profiles_cover_both_cache_layers() {
        let metadata =
            serde_yaml::from_str(include_str!("../examples/tiny-gpt2-tensorrt-metadata.json"))
                .unwrap();
        let profiles = GenerationProfiles::from_metadata(Some(&metadata)).unwrap();
        let encoded = profiles.for_model("model.onnx").unwrap();
        assert!(encoded.max.contains("past_key_values.1.key:1x2x16x64"));
        assert!(encoded.max.contains("past_key_values.1.value:1x2x16x64"));
        assert!(encoded
            .max
            .starts_with("attention_mask:1x32,input_ids:1x16,"));
    }

    #[test]
    fn explicit_profiles_encode_the_vast_prefill_and_decode_ranges() {
        let profiles = GenerationProfiles::from_metadata(Some(&metadata(valid_profile()))).unwrap();
        let encoded = profiles.for_model("onnx/model.onnx").unwrap();
        assert_eq!(encoded.min, "attention_mask:1x1,input_ids:1x1,past_key_values.0.key:1x2x0x64,past_key_values.0.value:1x2x0x64,position_ids:1x1");
        assert!(encoded.opt.starts_with("attention_mask:1x9,input_ids:1x9,"));
        assert!(encoded
            .max
            .starts_with("attention_mask:1x32,input_ids:1x16,"));
        assert_eq!(encoded, profiles.for_model("onnx/model.onnx").unwrap());
        assert!(profiles
            .for_model("other/model.onnx")
            .unwrap_err()
            .contains("no explicit"));
    }

    #[test]
    fn rejects_the_observed_inconsistent_implicit_profile() {
        let mut profile = valid_profile();
        profile["max"] = shapes(9, 9);
        profile["max"]["attention_mask"] = json!([1, 10]);
        let error = GenerationProfiles::from_metadata(Some(&metadata(profile)))
            .err()
            .unwrap();
        assert!(error.contains("sequence + past"), "{error}");
    }

    #[test]
    fn multiple_profiles_keep_input_order_and_model_options_isolated() {
        let first = ShapeProfile {
            min: serde_json::from_value(shapes(1, 0)).unwrap(),
            opt: serde_json::from_value(shapes(9, 0)).unwrap(),
            max: serde_json::from_value(shapes(16, 0)).unwrap(),
        };
        let second = ShapeProfile {
            min: serde_json::from_value(shapes(1, 0)).unwrap(),
            opt: serde_json::from_value(shapes(1, 9)).unwrap(),
            max: serde_json::from_value(shapes(1, 16)).unwrap(),
        };
        let profiles = GenerationProfiles {
            version: 1,
            models: BTreeMap::from([
                ("model.onnx".into(), vec![first, second.clone()]),
                ("decode/model.onnx".into(), vec![second]),
            ]),
        };
        profiles.validate().unwrap();
        let combined = profiles.for_model("model.onnx").unwrap();
        let decode = profiles.for_model("decode/model.onnx").unwrap();
        assert_eq!(combined.max.matches("input_ids:").count(), 2);
        assert!(combined.max.ends_with(&decode.max));
        assert_eq!(decode.max.matches("input_ids:").count(), 1);
        assert!(decode.max.starts_with("attention_mask:1x17,input_ids:1x1,"));
    }

    #[test]
    fn rejects_missing_versioned_configuration_and_unsafe_model_keys() {
        assert!(GenerationProfiles::from_metadata(None).is_err());
        let mut value = metadata(valid_profile());
        value["ort"]["tensorrt"]["version"] = serde_yaml::to_value(2).unwrap();
        assert!(GenerationProfiles::from_metadata(Some(&value)).is_err());
        for key in [
            "",
            "/model.onnx",
            "../model.onnx",
            "a/../model.onnx",
            "a//model.onnx",
            "./model.onnx",
            "a\\model.onnx",
            "C:/model.onnx",
        ] {
            assert!(validate_model_key(key).is_err(), "{key}");
        }
    }

    #[test]
    fn rejects_mismatched_inputs_ranks_bounds_and_profile_delimiters() {
        for replacement in [
            json!([]),
            json!([1]),
            json!([1, -1]),
            json!([1, 8]),
            json!([1, 2147483648_i64]),
        ] {
            let mut profile = valid_profile();
            profile["max"]["input_ids"] = replacement;
            assert!(GenerationProfiles::from_metadata(Some(&metadata(profile))).is_err());
        }
        let mut profile = valid_profile();
        profile["opt"].as_object_mut().unwrap().remove("input_ids");
        assert!(GenerationProfiles::from_metadata(Some(&metadata(profile))).is_err());
        for name in [
            "",
            "input:ids",
            "input,ids",
            "input;ids",
            "input\nids",
            "input ids",
        ] {
            let profile = json!({"min": {name: [1]}, "opt": {name: [1]}, "max": {name: [1]}});
            assert!(
                GenerationProfiles::from_metadata(Some(&metadata(profile))).is_err(),
                "{name}"
            );
        }
    }
}
