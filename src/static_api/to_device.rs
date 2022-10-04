use crate::{Buffer, Alloc, cpu::CPUPtr, PtrType, BufFlag, GraphReturn, Device, VecRead};

use super::{static_cpu, GPU};

impl<'a, T: Clone> Buffer<'a, T> {

    #[cfg(feature="cuda")]
    pub fn to_cuda(self) -> Buffer<'a, T, CUDA> {

    }

    #[cfg(feature="opencl")]
    pub fn to_cl(self) -> Buffer<'a, T, crate::OpenCL> {
        let device = super::static_opencl();
        let mut buf = Buffer::new(device, self.len);
        buf.write(&self);
        buf
    }

    #[cfg(feature="opencl")]
    #[cfg(not(feature="cuda"))]
    pub fn to_gpu(self) -> Buffer<'a, T, crate::OpenCL> {
        self.to_cl()
    }

    #[cfg(feature="cuda")]
    pub fn to_gpu(self) -> Buffer<'a, T, crate::CUDA> {
        self.to_cuda()
    }


}

impl<'a, T: Clone + Default, D: Device + VecRead<T, D> + GPU> Buffer<'a, T, D> {
    pub fn to_cpu(self) -> Buffer<'a, T> {
        Buffer::from((static_cpu(), self.read()))
    }
}

impl<'a, T: Clone> From<&[T]> for Buffer<'a, T> {
    fn from(slice: &[T]) -> Self {
        let device = static_cpu();
        Buffer {
            ptr: CPUPtr::from_ptrs(device.with_data(slice)),
            len: slice.len(),
            device: Some(device),
            flag: BufFlag::None,
            node: device.graph().add_leaf(slice.len()),
        }
    }
}

impl<'a, T: Clone, const N: usize> From<&[T; N]> for Buffer<'a, T> {
    fn from(slice: &[T; N]) -> Self {
        let device = static_cpu();
        Buffer {
            ptr: CPUPtr::from_ptrs(device.with_data(slice)),
            len: slice.len(),
            device: Some(device),
            flag: BufFlag::None,
            node: device.graph().add_leaf(slice.len()),
        }
    }
}

impl<'a, T: Clone, const N: usize> From<[T; N]> for Buffer<'a, T> {
    fn from(slice: [T; N]) -> Self {
        let device = static_cpu();
        Buffer {
            ptr: CPUPtr::from_ptrs(device.with_data(&slice)),
            len: slice.len(),
            device: Some(device),
            flag: BufFlag::None,
            node: device.graph().add_leaf(slice.len()),
        }
    }
}