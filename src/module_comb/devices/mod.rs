mod cpu;
pub use cpu::*;

mod cuda;
pub use cuda::*;

pub trait Device {}
