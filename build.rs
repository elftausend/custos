fn main() {
    #[cfg(feature = "opencl")]
    if has_device_unified_mem() {
        println!("cargo:rustc-cfg=unified_cl");
    }
}

#[cfg(feature = "opencl")]
fn has_device_unified_mem() -> bool {
    //TODO: idx as env var?
    custos::CLDevice::new(0)
        .expect("Could not get an OpenCL device.")
        .unified_mem()
}
