//! Manifest-selected input preprocessing for the native ORT adapter.
//!
//! The engine passes the client packet through ABI v1 without interpreting
//! backend-specific media. This module owns the model-side conversion from an
//! encoded image or PCM waveform to the tensor consumed by ONNX Runtime.

mod audio;
mod vision;

use crate::tensor::{BorrowedTensor, OwnedTensor};
use crate::{invalid_argument, FfiResult};
use audio::AudioPreprocessor;
use kapsl_core::Manifest;
use kapsl_engine_api::EngineModelInfo;
use vision::VisionPreprocessor;

pub(crate) enum InputPreprocessor {
    None,
    Vision(VisionPreprocessor),
    Audio(AudioPreprocessor),
}

impl InputPreprocessor {
    pub(crate) fn from_manifest(manifest: &Manifest) -> FfiResult<Self> {
        let Some(specification) = manifest
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("preprocess"))
        else {
            return Ok(Self::None);
        };
        let kind = specification
            .get("kind")
            .and_then(serde_yaml::Value::as_str)
            .ok_or_else(|| {
                invalid_argument("metadata.preprocess is set but missing a string `kind`")
            })?;
        match kind {
            "vision" => {
                let config = serde_yaml::from_value(specification.clone()).map_err(|error| {
                    invalid_argument(format!(
                        "decode metadata.preprocess vision configuration: {error}"
                    ))
                })?;
                Ok(Self::Vision(VisionPreprocessor::new(config)?))
            }
            "audio" => {
                let config = serde_yaml::from_value(specification.clone()).map_err(|error| {
                    invalid_argument(format!(
                        "decode metadata.preprocess audio configuration: {error}"
                    ))
                })?;
                Ok(Self::Audio(AudioPreprocessor::new(config)?))
            }
            other => Err(invalid_argument(format!(
                "unknown metadata.preprocess kind `{other}`"
            ))),
        }
    }

    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::None => "tensor",
            Self::Vision(_) => "vision",
            Self::Audio(_) => "audio",
        }
    }

    pub(crate) const fn is_identity(&self) -> bool {
        matches!(self, Self::None)
    }

    pub(crate) fn prepare(
        &self,
        inputs: &[BorrowedTensor<'_>],
    ) -> FfiResult<Option<PreparedInputs>> {
        let Some(primary) = inputs.first() else {
            return Err(invalid_argument(
                "native ORT preprocessing requires a primary input",
            ));
        };
        match self {
            Self::None => Ok(None),
            Self::Vision(preprocessor) => Ok(Some(PreparedInputs {
                primary: preprocessor.apply(primary)?,
                derived: Vec::new(),
            })),
            Self::Audio(preprocessor) => {
                let primary = preprocessor.apply(primary)?;
                let derived = preprocessor.derived_inputs(&primary)?;
                Ok(Some(PreparedInputs { primary, derived }))
            }
        }
    }

    pub(crate) fn planned_additional_bytes(
        &self,
        inputs: &[BorrowedTensor<'_>],
    ) -> FfiResult<usize> {
        let Some(primary) = inputs.first() else {
            return Err(invalid_argument(
                "native ORT preprocessing requires a primary input",
            ));
        };
        match self {
            Self::None => Ok(0),
            Self::Vision(preprocessor) => preprocessor.planned_bytes(primary),
            Self::Audio(preprocessor) => preprocessor.planned_bytes(primary),
        }
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        match self {
            Self::None | Self::Vision(_) => 0,
            Self::Audio(preprocessor) => preprocessor.resident_bytes(),
        }
    }

    pub(crate) fn validate_model_info(&self, info: &EngineModelInfo) -> FfiResult<()> {
        let expected_shape = match self {
            Self::None => return Ok(()),
            Self::Vision(preprocessor) => preprocessor.expected_shape(),
            Self::Audio(preprocessor) => preprocessor.expected_shape(),
        };
        validate_input_contract(info, 0, "input", "float32", &expected_shape)?;
        if let Self::Audio(preprocessor) = self {
            if let Some((name, dtype)) = preprocessor.length_contract() {
                let index = info
                    .input_names
                    .iter()
                    .position(|candidate| candidate == name)
                    .ok_or_else(|| {
                        invalid_argument(format!(
                            "audio preprocessing derives `{name}` but the ONNX model has no input with that name"
                        ))
                    })?;
                validate_input_contract(info, index, name, dtype, &[1])?;
            }
        }
        Ok(())
    }
}

fn validate_input_contract(
    info: &EngineModelInfo,
    index: usize,
    name: &str,
    expected_dtype: &str,
    expected_shape: &[i64],
) -> FfiResult<()> {
    let actual_dtype = info.input_dtypes.get(index).ok_or_else(|| {
        invalid_argument(format!(
            "ONNX model does not report a dtype for preprocessed input `{name}`"
        ))
    })?;
    if actual_dtype != expected_dtype {
        return Err(invalid_argument(format!(
            "preprocessed input `{name}` emits {expected_dtype}, but the ONNX model requires {actual_dtype}"
        )));
    }
    let actual_shape = info.input_shapes.get(index).ok_or_else(|| {
        invalid_argument(format!(
            "ONNX model does not report a shape for preprocessed input `{name}`"
        ))
    })?;
    if actual_shape.len() != expected_shape.len()
        || actual_shape
            .iter()
            .zip(expected_shape)
            .any(|(actual, expected)| *actual > 0 && *expected > 0 && actual != expected)
    {
        return Err(invalid_argument(format!(
            "preprocessed input `{name}` emits shape {expected_shape:?}, but the ONNX model requires {actual_shape:?}"
        )));
    }
    Ok(())
}

pub(crate) struct PreparedInputs {
    primary: OwnedTensor,
    derived: Vec<OwnedTensor>,
}

impl PreparedInputs {
    pub(crate) fn views<'a, 'b>(
        &'a self,
        original: &'a [BorrowedTensor<'b>],
    ) -> Vec<BorrowedTensor<'a>>
    where
        'b: 'a,
    {
        let mut views = Vec::with_capacity(original.len().saturating_add(self.derived.len()));
        views.push(self.primary.as_borrowed());
        for input in original.iter().skip(1) {
            if !self
                .derived
                .iter()
                .any(|derived| derived.name == input.name)
            {
                views.push(BorrowedTensor {
                    name: input.name,
                    dtype: input.dtype,
                    shape: input.shape,
                    data: input.data,
                });
            }
        }
        views.extend(self.derived.iter().map(OwnedTensor::as_borrowed));
        views
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kapsl_backend_abi::{KAPSL_DTYPE_F32, KAPSL_DTYPE_I64, KAPSL_DTYPE_U8};

    #[test]
    fn prepared_views_replace_primary_and_manifest_owned_inputs() {
        let original_primary_shape = [3_i64];
        let original_primary_data = [1_u8, 2, 3];
        let stale_length_shape = [1_i64];
        let stale_length_data = 99_i64.to_ne_bytes();
        let retained_shape = [1_i64];
        let retained_data = 7_i64.to_ne_bytes();
        let originals = [
            BorrowedTensor {
                name: "input",
                dtype: KAPSL_DTYPE_U8,
                shape: &original_primary_shape,
                data: &original_primary_data,
            },
            BorrowedTensor {
                name: "length",
                dtype: KAPSL_DTYPE_I64,
                shape: &stale_length_shape,
                data: &stale_length_data,
            },
            BorrowedTensor {
                name: "retained",
                dtype: KAPSL_DTYPE_I64,
                shape: &retained_shape,
                data: &retained_data,
            },
        ];
        let prepared = PreparedInputs {
            primary: OwnedTensor {
                name: "input".to_string(),
                dtype: KAPSL_DTYPE_F32,
                shape: vec![1, 1],
                data: 1.0_f32.to_ne_bytes().to_vec(),
            },
            derived: vec![OwnedTensor {
                name: "length".to_string(),
                dtype: KAPSL_DTYPE_I64,
                shape: vec![1],
                data: 1_i64.to_ne_bytes().to_vec(),
            }],
        };
        let views = prepared.views(&originals);
        assert_eq!(views.len(), 3);
        assert_eq!(views[0].dtype, KAPSL_DTYPE_F32);
        assert_eq!(views[1].name, "retained");
        assert_eq!(views[2].name, "length");
        assert_eq!(views[2].data, 1_i64.to_ne_bytes());
    }
}
