mod cpu;
pub use cpu::*;

mod cuda;
pub use cuda::*;

use super::{Alloc, OnDropBuffer};

pub trait Device: Alloc + OnDropBuffer {
    type Error;

    #[inline]
    fn new() -> Result<Self, Self::Error> {
        todo!()
    }
}
