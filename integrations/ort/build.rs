fn main() {
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libkapsl_backend_ort.dylib");
}
