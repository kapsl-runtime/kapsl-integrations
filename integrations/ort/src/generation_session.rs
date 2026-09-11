//! Bind TensorRT session options to the model replica that supplied them.

use crate::tensorrt_profiles::{GenerationProfiles, ProviderProfileStrings};
use kapsl_engine_api::EngineError;
use kapsl_llm::onnx_session::{OnnxSessionConfigurator, OnnxSessionContext};
use ort::ep::{TensorRT, CUDA};
use ort::session::builder::SessionBuilder;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub(crate) struct GenerationSessionConfigurator {
    profiles: Arc<GenerationProfiles>,
    model_root: PathBuf,
    device_id: i32,
}

impl OnnxSessionConfigurator for GenerationSessionConfigurator {
    fn configure(
        &self,
        builder: SessionBuilder,
        context: OnnxSessionContext<'_>,
    ) -> Result<SessionBuilder, EngineError> {
        let profiles = self
            .profiles_for(context.model_path, context.provider, context.device_id)
            .map_err(EngineError::backend)?;
        builder
            .with_execution_providers([
                TensorRT::default()
                    .with_device_id(self.device_id)
                    .with_profile_min_shapes(profiles.min)
                    .with_profile_opt_shapes(profiles.opt)
                    .with_profile_max_shapes(profiles.max)
                    .build()
                    .error_on_failure(),
                CUDA::default()
                    .with_device_id(self.device_id)
                    .build()
                    .error_on_failure(),
            ])
            .map_err(|error| {
                EngineError::backend(format!("configure TensorRT generation providers: {error}"))
            })
    }
}

impl GenerationSessionConfigurator {
    pub(crate) fn new(
        profiles: Arc<GenerationProfiles>,
        model_root: &Path,
        device_id: i32,
    ) -> Result<Self, String> {
        if device_id < 0 {
            return Err("TensorRT generation requires a nonnegative device ID".into());
        }
        let model_root = model_root
            .canonicalize()
            .map_err(|error| format!("resolve TensorRT model root: {error}"))?;
        if !model_root.is_dir() {
            return Err("TensorRT generation model root must be a directory".into());
        }
        Ok(Self {
            profiles,
            model_root,
            device_id,
        })
    }

    pub(crate) fn validate_model(&self, model_path: &Path) -> Result<(), String> {
        self.profiles_for(model_path, "tensorrt", self.device_id)
            .map(|_| ())
    }

    fn profiles_for(
        &self,
        model_path: &Path,
        provider: &str,
        device_id: i32,
    ) -> Result<ProviderProfileStrings, String> {
        if provider != "tensorrt" || device_id != self.device_id {
            return Err(
                "TensorRT session provider/device does not match its adapter instance".into(),
            );
        }
        let model_path = model_path
            .canonicalize()
            .map_err(|error| format!("resolve TensorRT model file: {error}"))?;
        if !model_path.is_file() {
            return Err("TensorRT generation requires a regular ONNX model file".into());
        }
        let relative_path = model_path
            .strip_prefix(&self.model_root)
            .map_err(|_| "TensorRT model file is outside its configured model root".to_string())?;
        let key = relative_path
            .components()
            .map(|part| {
                part.as_os_str()
                    .to_str()
                    .ok_or_else(|| "TensorRT model profile path must be UTF-8".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?
            .join("/");
        self.profiles.for_model(&key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn configuration(root: &Path, max: u32) -> GenerationSessionConfigurator {
        let metadata = serde_yaml::to_value(json!({"ort": {"tensorrt": {
            "version": 1, "models": {
                "model.onnx": [{"min": {"tokens": [1,1]}, "opt": {"tokens": [1,1]}, "max": {"tokens": [1,max]}}],
                "decode/model.onnx": [{"min": {"tokens": [1,1]}, "opt": {"tokens": [1,1]}, "max": {"tokens": [1,1]}}]
            }
        }}})).unwrap();
        let profiles = GenerationProfiles::from_metadata(Some(&metadata)).unwrap();
        GenerationSessionConfigurator::new(Arc::new(profiles), root, 2).unwrap()
    }

    fn model_files() -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("decode")).unwrap();
        std::fs::write(root.path().join("model.onnx"), b"fixture").unwrap();
        std::fs::write(root.path().join("decode/model.onnx"), b"fixture").unwrap();
        root
    }

    #[test]
    fn model_stage_device_and_reload_options_stay_with_their_instance() {
        let first_root = model_files();
        let second_root = model_files();
        let first = configuration(first_root.path(), 16);
        let second = configuration(second_root.path(), 32);
        let first_model = first_root.path().join("model.onnx");
        first.validate_model(&first_model).unwrap();
        assert!(first
            .validate_model(&second_root.path().join("model.onnx"))
            .is_err());
        assert_eq!(
            first.profiles_for(&first_model, "tensorrt", 2).unwrap().max,
            "tokens:1x16"
        );
        assert_eq!(
            second
                .profiles_for(&second_root.path().join("model.onnx"), "tensorrt", 2)
                .unwrap()
                .max,
            "tokens:1x32"
        );
        assert_eq!(
            first
                .profiles_for(&first_root.path().join("decode/model.onnx"), "tensorrt", 2)
                .unwrap()
                .max,
            "tokens:1x1"
        );
        assert_eq!(
            first.profiles_for(&first_model, "tensorrt", 2).unwrap().max,
            "tokens:1x16"
        );
        assert!(first
            .profiles_for(&second_root.path().join("model.onnx"), "tensorrt", 2)
            .is_err());
        assert!(first.profiles_for(&first_model, "cuda", 2).is_err());
        assert!(first.profiles_for(&first_model, "tensorrt", 3).is_err());
    }

    #[test]
    fn missing_stage_profiles_never_reuse_a_matching_basename() {
        let root = model_files();
        let configuration = configuration(root.path(), 16);
        std::fs::create_dir(root.path().join("prefill")).unwrap();
        let model = root.path().join("prefill/model.onnx");
        std::fs::write(&model, b"fixture").unwrap();
        let error = configuration
            .profiles_for(&model, "tensorrt", 2)
            .unwrap_err();
        assert!(error.contains("no explicit"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_cannot_select_model_profiles_outside_the_replica_root() {
        let root = model_files();
        let foreign = model_files();
        let configuration = configuration(root.path(), 16);
        let model = root.path().join("model.onnx");
        std::fs::remove_file(&model).unwrap();
        std::os::unix::fs::symlink(foreign.path().join("model.onnx"), &model).unwrap();
        assert!(configuration
            .profiles_for(&model, "tensorrt", 2)
            .unwrap_err()
            .contains("outside"));
    }
}
