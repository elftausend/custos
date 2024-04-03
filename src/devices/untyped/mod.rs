pub mod storages;
mod untyped_device;

#[cfg(not(feature = "cuda"))]
mod dummy_cuda;

#[cfg(not(feature = "cuda"))]
pub use dummy_cuda::*;

mod ops;
mod matches_type;
pub use matches_type::*;
