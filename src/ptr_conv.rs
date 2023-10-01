use crate::{flag::AllocFlag, Device, Shape};

pub trait PtrConv<D: Device = Self>: Device {
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(
        data: &Self::Data<T, IS>,
        flag: AllocFlag,
    ) -> D::Data<Conv, OS>;
}
