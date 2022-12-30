//#[cfg(not(target_os = "macos"))]
#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "opt-cache")]
mod graph;

mod demo_impl;
mod threading;

#[cfg(feature = "opencl")]
#[test]
fn test_debug_fmt_cl_dev() -> custos::Result<()> {
    use custos::OpenCL;

    let device = OpenCL::new(0)?;
    println!("device: {device:?}");
    Ok(())
}
