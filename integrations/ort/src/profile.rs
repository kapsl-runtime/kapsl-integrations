use crate::{invalid_argument, FfiResult};
use kapsl_backend_abi::{
    KAPSL_BACKEND_CAP_BATCHING, KAPSL_BACKEND_CAP_CANCELLATION,
    KAPSL_BACKEND_CAP_CONCURRENT_INFERENCE, KAPSL_BACKEND_CAP_CPU, KAPSL_BACKEND_CAP_CUDA,
    KAPSL_BACKEND_CAP_GOVERNED_DEVICE_ALLOCATOR, KAPSL_BACKEND_CAP_MEMORY_REPORTING,
    KAPSL_BACKEND_CAP_STREAMING, KAPSL_BACKEND_CAP_TENSORRT,
};

const COMMON_CAPABILITIES: u64 = KAPSL_BACKEND_CAP_BATCHING
    | KAPSL_BACKEND_CAP_CANCELLATION
    | KAPSL_BACKEND_CAP_MEMORY_REPORTING
    | KAPSL_BACKEND_CAP_CONCURRENT_INFERENCE
    | KAPSL_BACKEND_CAP_STREAMING;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProviderProfile {
    Cpu,
    Cuda12,
    TensorRt10,
}

impl ProviderProfile {
    pub(crate) const ALL: [Self; 3] = [Self::Cpu, Self::Cuda12, Self::TensorRt10];

    pub(crate) const fn pack_profile(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda12 => "cuda12",
            Self::TensorRt10 => "tensorrt10",
        }
    }

    pub(crate) const fn accelerator_profile(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda12 => "cuda",
            Self::TensorRt10 => "tensorrt",
        }
    }

    pub(crate) const fn provider(self) -> &'static str {
        self.accelerator_profile()
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda12 => "CUDA 12",
            Self::TensorRt10 => "TensorRT 10",
        }
    }

    pub(crate) const fn requires_governed_device_memory(self) -> bool {
        !matches!(self, Self::Cpu)
    }

    pub(crate) const fn supports_generation(self) -> bool {
        matches!(self, Self::Cpu)
    }

    pub(crate) const fn capabilities(self) -> u64 {
        let execution = match self {
            Self::Cpu => KAPSL_BACKEND_CAP_CPU,
            Self::Cuda12 => KAPSL_BACKEND_CAP_CUDA | KAPSL_BACKEND_CAP_GOVERNED_DEVICE_ALLOCATOR,
            Self::TensorRt10 => {
                KAPSL_BACKEND_CAP_CUDA
                    | KAPSL_BACKEND_CAP_TENSORRT
                    | KAPSL_BACKEND_CAP_GOVERNED_DEVICE_ALLOCATOR
            }
        };
        COMMON_CAPABILITIES | execution
    }

    pub(crate) fn validate_contract(
        self,
        pack_profile: &str,
        provider: &str,
        accelerator_profile: &str,
        require_governed_device_memory: u32,
        host_has_governed_callbacks: bool,
    ) -> FfiResult<()> {
        if pack_profile != self.pack_profile() {
            return Err(invalid_argument(format!(
                "ORT {} adapter cannot initialize signed pack profile `{pack_profile}`; expected `{}`",
                self.label(),
                self.pack_profile()
            )));
        }
        if !provider.eq_ignore_ascii_case(self.provider()) {
            return Err(invalid_argument(format!(
                "ORT {} adapter requires provider `{}`, received `{provider}`",
                self.label(),
                self.provider()
            )));
        }
        if accelerator_profile != self.accelerator_profile() {
            return Err(invalid_argument(format!(
                "ORT {} adapter requires accelerator profile `{}`, received `{accelerator_profile}`",
                self.label(),
                self.accelerator_profile()
            )));
        }

        let require_governed = match require_governed_device_memory {
            0 => false,
            1 => true,
            other => {
                return Err(invalid_argument(format!(
                    "native ORT governed-device-memory flag must be 0 or 1, received {other}"
                )))
            }
        };
        if require_governed != self.requires_governed_device_memory() {
            return Err(invalid_argument(format!(
                "ORT {} adapter requires governed device memory to be {}",
                self.label(),
                self.requires_governed_device_memory()
            )));
        }
        if self.requires_governed_device_memory() && !host_has_governed_callbacks {
            return Err(invalid_argument(format!(
                "ORT {} adapter requires allocate, free, and synchronize device callbacks",
                self.label()
            )));
        }
        Ok(())
    }
}

#[cfg(feature = "profile-cpu")]
pub(crate) const COMPILED_PROFILE: ProviderProfile = ProviderProfile::Cpu;
#[cfg(feature = "profile-cuda12")]
pub(crate) const COMPILED_PROFILE: ProviderProfile = ProviderProfile::Cuda12;
#[cfg(feature = "profile-tensorrt10")]
pub(crate) const COMPILED_PROFILE: ProviderProfile = ProviderProfile::TensorRt10;

#[cfg(test)]
mod tests {
    use super::*;
    use kapsl_backend_abi::{
        KAPSL_BACKEND_CAP_EXECUTION_MASK, KAPSL_BACKEND_CAP_GOVERNED_DEVICE_ALLOCATOR,
    };

    #[test]
    fn every_profile_has_one_exact_contract() {
        for profile in ProviderProfile::ALL {
            assert!(profile
                .validate_contract(
                    profile.pack_profile(),
                    profile.provider(),
                    profile.accelerator_profile(),
                    u32::from(profile.requires_governed_device_memory()),
                    profile.requires_governed_device_memory(),
                )
                .is_ok());
            assert!(profile
                .validate_contract(
                    "wrong",
                    profile.provider(),
                    profile.accelerator_profile(),
                    u32::from(profile.requires_governed_device_memory()),
                    profile.requires_governed_device_memory(),
                )
                .is_err());
            assert!(profile
                .validate_contract(
                    profile.pack_profile(),
                    "wrong",
                    profile.accelerator_profile(),
                    u32::from(profile.requires_governed_device_memory()),
                    profile.requires_governed_device_memory(),
                )
                .is_err());
            assert!(profile
                .validate_contract(
                    profile.pack_profile(),
                    profile.provider(),
                    "wrong",
                    u32::from(profile.requires_governed_device_memory()),
                    profile.requires_governed_device_memory(),
                )
                .is_err());
        }
    }

    #[test]
    fn accelerator_profiles_require_the_complete_governed_host_contract() {
        for profile in [ProviderProfile::Cuda12, ProviderProfile::TensorRt10] {
            assert!(profile
                .validate_contract(
                    profile.pack_profile(),
                    profile.provider(),
                    profile.accelerator_profile(),
                    0,
                    true,
                )
                .is_err());
            assert!(profile
                .validate_contract(
                    profile.pack_profile(),
                    profile.provider(),
                    profile.accelerator_profile(),
                    1,
                    false,
                )
                .is_err());
        }
        assert!(ProviderProfile::Cpu
            .validate_contract("cpu", "cpu", "cpu", 1, true)
            .is_err());
        assert!(ProviderProfile::Cpu
            .validate_contract("cpu", "cpu", "cpu", 2, true)
            .is_err());
    }

    #[test]
    fn capabilities_are_profile_specific_and_consistent() {
        assert!(ProviderProfile::Cpu.supports_generation());
        assert!(!ProviderProfile::Cuda12.supports_generation());
        assert!(!ProviderProfile::TensorRt10.supports_generation());
        assert_eq!(
            ProviderProfile::Cpu.capabilities() & KAPSL_BACKEND_CAP_EXECUTION_MASK,
            KAPSL_BACKEND_CAP_CPU
        );
        assert_eq!(
            ProviderProfile::Cuda12.capabilities() & KAPSL_BACKEND_CAP_EXECUTION_MASK,
            KAPSL_BACKEND_CAP_CUDA
        );
        assert_eq!(
            ProviderProfile::TensorRt10.capabilities() & KAPSL_BACKEND_CAP_EXECUTION_MASK,
            KAPSL_BACKEND_CAP_CUDA | KAPSL_BACKEND_CAP_TENSORRT
        );
        assert_eq!(
            ProviderProfile::Cpu.capabilities() & KAPSL_BACKEND_CAP_GOVERNED_DEVICE_ALLOCATOR,
            0
        );
        assert_ne!(
            ProviderProfile::Cuda12.capabilities() & KAPSL_BACKEND_CAP_GOVERNED_DEVICE_ALLOCATOR,
            0
        );
        assert_ne!(
            ProviderProfile::TensorRt10.capabilities()
                & KAPSL_BACKEND_CAP_GOVERNED_DEVICE_ALLOCATOR,
            0
        );

        // SAFETY: the exported entrypoint returns a process-lifetime table.
        let api = unsafe { &*crate::kapsl_backend_v1() };
        assert!(api.capabilities_are_consistent());
        assert_eq!(api.capabilities, COMPILED_PROFILE.capabilities());
    }
}
