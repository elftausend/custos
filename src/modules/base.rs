use core::any::Any;

use crate::{
    flag::AllocFlag, AddOperation, Alloc, Buffer, Device, HashLocation, Module, Parents, Retrieve,
    Setup, Shape,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Base;

impl<D> Module<D> for Base {
    type Module = Base;

    #[inline]
    fn new() -> Self::Module {
        Base
    }
}

impl AddOperation for Base {
    #[inline]
    unsafe fn add_operation<T: 'static, D: Device + 'static, S: Shape>(
        &self,
        out: &mut Buffer<T, D, S>,
        operation: impl Fn(&mut dyn Any),
    ) {
        let out: &mut Buffer<T, D, S> = unsafe { core::mem::transmute(out) };
        operation(out);
    }

    #[inline]
    fn add_operation2(&self, mut operation: impl crate::Operation) {
        operation.forward()
    }
}

impl<D> Setup<D> for Base {}

impl<D, T, S: Shape> Retrieve<D, T, S> for Base {
    #[inline]
    fn retrieve<const NUM_PARENTS: usize>(
        &self,
        device: &D,
        _parents: impl Parents<NUM_PARENTS>,
        alloc_fn: impl FnOnce(&D, AllocFlag) -> D::Data<T, S>,
    ) -> <D>::Data<T, S>
    where
        S: crate::Shape,
        D: Alloc<T>,
    {
        alloc_fn(device, AllocFlag::None)
    }
}

impl crate::UseGpuOrCpu for Base {
    #[inline]
    fn use_cpu_or_gpu(
        &self,
        _location: HashLocation,
        _input_lengths: &[usize],
        _cpu_op: impl FnMut(),
        mut gpu_op: impl FnMut(),
    ) -> crate::GpuOrCpuInfo {
        gpu_op();
        crate::GpuOrCpuInfo {
            use_cpu: false,
            is_result_cached: false,
        }
    }
}

#[cfg(feature = "autograd")]
impl crate::TapeActions for Base {}
