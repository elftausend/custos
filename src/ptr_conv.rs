use core::mem::{align_of, size_of};

use crate::{backend::Backend, cpu::CPUPtr, flag::AllocFlag, Device, OnDropBuffer, Shape};

use super::CPU;

pub trait PtrConv<D: Device = Self>: Device {
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(
        data: &Self::Data<T, IS>,
        flag: AllocFlag,
    ) -> D::Data<Conv, OS>;
}

// impl for all devices
impl<Mods: OnDropBuffer, OtherMods> PtrConv<CPU<OtherMods>> for CPU<Mods> {
    #[inline]
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(
        data: &CPUPtr<T>,
        flag: AllocFlag,
    ) -> CPUPtr<Conv> {
        CPUPtr {
            ptr: data.ptr as *mut Conv,
            len: data.len,
            flag,
            align: Some(align_of::<T>()),
            size: Some(size_of::<T>()),
        }
    }
}

impl<Mods: OnDropBuffer, OtherMods: OnDropBuffer> PtrConv<Backend<CPU, OtherMods>> for CPU<Mods> {
    #[inline]
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(
        data: &CPUPtr<T>,
        flag: AllocFlag,
    ) -> CPUPtr<Conv> {
        CPUPtr {
            ptr: data.ptr as *mut Conv,
            len: data.len,
            flag,
            align: Some(align_of::<T>()),
            size: Some(size_of::<T>()),
        }
    }
}
