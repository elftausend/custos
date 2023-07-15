use crate::Shape;

use super::{Alloc, Base, CPU};

pub struct Buffer<'a, T = f32, D: Alloc = CPU<Base>, S: Shape = ()> {
    /// the type of pointer
    pub data: D::Data<T, S>,
    /// A reference to the corresponding device. Mainly used for operations without a device parameter.
    pub device: Option<&'a D>,
}
