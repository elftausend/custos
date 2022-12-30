
mod element_wise;
mod transpose;

pub use element_wise::*;

mod cpu;

#[cfg(feature = "opencl")]
mod opencl;

#[cfg(feature = "wgpu")]
mod wgpu;
