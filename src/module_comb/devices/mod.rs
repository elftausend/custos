mod cpu;
pub use cpu::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

use super::{Alloc, OnDropBuffer};


