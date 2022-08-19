//#[cfg(not(target_os = "macos"))]
#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "metal")]
mod metal;

#[cfg(feature = "opencl")]
#[test]
fn test_debug_fmt_cl_dev() -> custos::Result<()> {
    use custos::CLDevice;

    let device = CLDevice::new(0)?;
    println!("device: {device:?}");
    Ok(())
}
