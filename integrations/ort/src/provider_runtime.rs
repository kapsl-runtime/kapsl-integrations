//! Bind each accelerator adapter to the runtime beside its signed entrypoint.
//! No ORT_DYLIB_PATH, system-library search or embedded-runtime fallback.

#[cfg(not(test))]
use std::ffi::CStr;
use std::path::PathBuf;
use std::sync::OnceLock;

const GOVERNANCE_MARKER: &str = "; kapsl-scoped-providers-v1";

#[cfg(any(test, target_os = "linux"))]
pub(crate) fn library_name() -> String {
    #[cfg(feature = "profile-cuda12")]
    let profile = "cuda12";
    #[cfg(not(feature = "profile-cuda12"))]
    let profile = "tensorrt10";
    format!(
        "libkapsl_ort_{profile}_scoped_v1_a{}.so",
        env!("CARGO_PKG_VERSION").replace('.', "_")
    )
}

#[cfg(all(target_os = "linux", not(test)))]
fn runtime_path() -> Result<PathBuf, String> {
    // A private function address belongs to this adapter, unlike current_exe
    // (the engine) or the process's possibly interposed OrtGetApiBase symbol.
    fn anchor() {}
    let mut info = std::mem::MaybeUninit::<libc::Dl_info>::uninit();
    // SAFETY: dladdr accepts a function address and writes a complete Dl_info
    // on success. dli_fname is owned by the loader while this adapter is live.
    unsafe {
        if libc::dladdr(
            anchor as *const () as *const libc::c_void,
            info.as_mut_ptr(),
        ) == 0
        {
            return Err("locate signed ORT adapter library".into());
        }
        let info = info.assume_init();
        if info.dli_fname.is_null() {
            return Err("ORT adapter has no library path".into());
        }
        use std::os::unix::ffi::OsStrExt;
        let path = std::path::Path::new(std::ffi::OsStr::from_bytes(
            CStr::from_ptr(info.dli_fname).to_bytes(),
        ));
        let canonical = path
            .canonicalize()
            .map_err(|e| format!("resolve ORT adapter path: {e}"))?;
        let root = canonical
            .parent()
            .ok_or("ORT adapter has no pack directory")?;
        let runtime = root.join(library_name());
        let metadata = runtime.symlink_metadata().map_err(|e| {
            format!(
                "signed pack is missing governed ORT runtime {}: {e}",
                runtime.display()
            )
        })?;
        if !metadata.file_type().is_file() {
            return Err("governed ORT runtime must be a regular file in its signed pack".into());
        }
        Ok(runtime)
    }
}

#[cfg(all(not(target_os = "linux"), not(test)))]
fn runtime_path() -> Result<PathBuf, String> {
    Err("this accelerator pack requires a compatible Linux scoped ORT runtime; use a compatible signed pack for this platform".into())
}

#[cfg(test)]
fn runtime_path() -> Result<PathBuf, String> {
    // Host-only tests deliberately load the verified CPU test runtime and
    // exercise fake device callbacks. This path does not exist in pack builds.
    let directory =
        std::env::var_os("ORT_LIB_LOCATION").ok_or("host tests require ORT_LIB_LOCATION")?;
    let name = if cfg!(target_os = "macos") {
        "libonnxruntime.1.23.2.dylib"
    } else if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else {
        "libonnxruntime.so.1"
    };
    Ok(PathBuf::from(directory).join(name))
}

fn verify_build_info(info: &str) -> Result<(), String> {
    if info.ends_with(GOVERNANCE_MARKER) {
        Ok(())
    } else {
        Err("incompatible ORT runtime: scoped provider governance v1 is required; embedded/stock ORT cannot serve this accelerator pack".into())
    }
}

pub(crate) fn initialize() -> Result<(), String> {
    static RUNTIME: OnceLock<Result<libloading::Library, String>> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        let path = runtime_path()?;
        // SAFETY: the generic host verified this signed pack before entrypoint
        // loading. The private core and provider SONAMEs/symbols are namespaced
        // per profile and bind internal symbols locally at integration build.
        let library = unsafe { libloading::Library::new(&path) }
            .map_err(|e| format!("load governed ORT runtime {}: {e}", path.display()))?;
        // SAFETY: the pinned runtime exports ORT's documented C API. Retain its
        // library handle for the lifetime of all copied function pointers.
        unsafe {
            let getter = library.get::<unsafe extern "C" fn() -> *const ort::sys::OrtApiBase>(b"OrtGetApiBase\0")
                .map_err(|e| format!("governed ORT API entrypoint: {e}"))?;
            let base = getter().as_ref().ok_or("governed ORT returned a null API base")?;
            let api = (base.GetApi)(ort::sys::ORT_API_VERSION).as_ref().ok_or("governed ORT does not support the adapter's API version")?;
            #[cfg(not(test))]
            {
                let info = (api.GetBuildInfoString)();
                if info.is_null() { return Err("governed ORT returned no build identity".into()); }
                verify_build_info(CStr::from_ptr(info).to_str().map_err(|_| "ORT build identity is not UTF-8")?)?;
            }
            if !ort::set_api(*api) { return Err("ORT API was initialized before the signed pack runtime; refusing runtime substitution".into()); }
        }
        Ok(library)
    }).as_ref().map(|_| ()).map_err(Clone::clone)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stock_or_embedded_builds_cannot_satisfy_the_provider_contract() {
        assert!(verify_build_info("ORT Build Info: git-commit-id=a83fc4d").is_err());
        assert!(verify_build_info("ORT Build Info; kapsl-scoped-providers-v0").is_err());
        assert!(verify_build_info(&format!("ORT Build Info{GOVERNANCE_MARKER}")).is_ok());
        assert!(library_name().starts_with("libkapsl_ort_"));
    }
}
