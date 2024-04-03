pub mod storages;
mod untyped_device;

#[cfg(not(feature = "cuda"))]
mod dummy_cuda;

#[cfg(not(feature = "cuda"))]
pub use dummy_cuda::*;
