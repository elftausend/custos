pub mod libs;
mod buffer;

//pub use libs::*;
pub use buffer::*;

pub trait VecRead<T>: Device {
    fn read(&self, buf: &Buffer<T>) -> Vec<T>;
}