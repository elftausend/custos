use crate::{cpu::CPUPtr, flag::AllocFlag, Device, Shape};

pub trait PtrConv<D: Device = Self>: Device {
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(
        data: &Self::Data<T, IS>,
        flag: AllocFlag,
    ) -> D::Data<Conv, OS>;
}

pub trait ConvPtr<NewT, NewS: Shape> {
    type ConvertTo;
    unsafe fn convert(&self, flag: AllocFlag) -> Self::ConvertTo;
}

impl<T, NewT, NewS: Shape> ConvPtr<NewT, NewS> for crate::cpu::CPUPtr<T> {
    type ConvertTo = CPUPtr<NewT>;

    unsafe fn convert(&self, flag: AllocFlag) -> Self::ConvertTo {
        CPUPtr {
            ptr: self.ptr as *mut NewT,
            len: self.len,
            flag,
            align: Some(core::mem::align_of::<T>()),
            size: Some(core::mem::size_of::<T>()),
        }
    }
}
