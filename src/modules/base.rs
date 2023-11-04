use crate::{
    flag::AllocFlag, AddOperation, Alloc, Buffer, Device, ExecNow, HashLocation, Module,
    OnDropBuffer, OnNewBuffer, OptimizeMemGraph, Parents, Retrieve, Setup, Shape,
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
    fn add_op<S: Shape, Args: Parents<N>, const N: usize>(
        &self,
        mut args: Args,
        out: &mut Buffer<T, D, S>,
        operation: fn(&mut Buffer<T, D, S>, &mut Args) -> crate::Result<()>,
    ) -> crate::Result<()> {
        operation(out, &mut args)
    }

    #[inline]
    fn ops_count(&self) -> usize {
        0
    }
}

impl<D: Device> ExecNow<D> for Base {
    #[inline]
    fn exec_now(&self, _range_bounds: impl core::ops::RangeBounds<usize>) -> crate::Result<()> {
        Ok(())
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

impl OptimizeMemGraph for Base {
    #[inline]
    fn optimize_mem_graph(
        &self,
        _cache_traces: Option<&[crate::TranslatedCacheTrace]>,
    ) -> crate::Result<()> {
        Ok(())
    }
}

#[cfg(feature = "autograd")]
impl crate::TapeActions for Base {}
