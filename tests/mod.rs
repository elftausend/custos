//#[cfg(not(target_os = "macos"))]
#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "opencl")]
mod opencl;

#[cfg(feature = "vulkan")]
mod vk;

#[cfg(feature = "graph")]
mod graph;

mod demo_impl;
mod threading;

#[cfg(feature = "opencl")]
#[test]
fn test_debug_fmt_cl_dev() -> custos::Result<()> {
    use custos::{Base, OpenCL, prelude::chosen_cl_idx};

    let device = OpenCL::<Base>::new(chosen_cl_idx())?;
    println!("device: {device:?}");
    Ok(())
}
