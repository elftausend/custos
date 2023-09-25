use crate::{
    flag::AllocFlag, AddOperation, Alloc, Buffer, Device, HashLocation, Module, OnDropBuffer,
    OnNewBuffer, Parents, Retrieve, Setup, Shape,
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

impl<T, D: Device> AddOperation<T, D> for Base {
    #[inline]
    unsafe fn add_operation<S: Shape>(
        &self,
        out: &mut Buffer<T, D, S>,
        operation: impl Fn(&mut Buffer<T, D, S>),
    ) {
        operation(out);
    }
}

impl<D> Setup<D> for Base {}

impl<T, D: Device, S: Shape> OnNewBuffer<T, D, S> for Base {}

impl OnDropBuffer for Base {}

impl<D, T> Retrieve<D, T> for Base {
    #[inline]
    fn retrieve<S, const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        _parents: impl Parents<NUM_PARENTS>,
    ) -> <D>::Data<T, S>
    where
        S: crate::Shape,
        D: Alloc<T>,
    {
        device.alloc(len, AllocFlag::None)
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
