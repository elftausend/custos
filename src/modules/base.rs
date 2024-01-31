use crate::{
    flag::AllocFlag, AddGradFn, AddOperation, Alloc, Device, ExecNow, HasId, HashLocation, Module,
    OnDropBuffer, OnNewBuffer, Parents, PtrType, Retrieve, Setup, Shape, WrappedData,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Base;

impl WrappedData for Base {
    type Wrap<T, Base: HasId + PtrType> = Base;

    #[inline]
    fn wrap_in_base<T, Base: HasId + PtrType>(&self, base: Base) -> Self::Wrap<T, Base> {
        base
    }

    #[inline]
    fn wrapped_as_base<T, Base: HasId + PtrType>(wrap: &Self::Wrap<T, Base>) -> &Base {
        wrap
    }

    #[inline]
    fn wrapped_as_base_mut<T, Base: HasId + PtrType>(wrap: &mut Self::Wrap<T, Base>) -> &mut Base {
        wrap
    }
}

impl<D: Device> Module<D> for Base {
    type Module = Base;

    #[inline]
    fn new() -> Self::Module {
        Base
    }
}

impl AddOperation for Base {
    #[inline]
    fn ops_count(&self) -> usize {
        0
    }

    fn add_op<Args: Parents<N>, const N: usize>(
        &self,
        mut args: Args,
        operation: fn(&mut Args) -> crate::Result<()>,
    ) -> crate::Result<()> {
        operation(&mut args)
    }
}

impl<D: Device> ExecNow<D> for Base {
    #[inline]
    fn exec_now(
        &self,
        _device: &D,
        _range_bounds: impl core::ops::RangeBounds<usize>,
    ) -> crate::Result<()> {
        Ok(())
    }
}

impl<D> Setup<D> for Base {}

impl<T, D: Device, S: Shape> OnNewBuffer<T, D, S> for Base {}

impl OnDropBuffer for Base {}

impl<D, T, S: Shape> Retrieve<D, T, S> for Base {
    #[inline]
    unsafe fn retrieve<const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        _parents: impl Parents<NUM_PARENTS>,
    ) -> Self::Wrap<T, D::Base<T, S>>
    where
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

#[cfg(feature = "graph")]
impl crate::OptimizeMemGraph for Base {
    #[inline]
    fn optimize_mem_graph<D: 'static>(
        &self,
        _device: &D,
        _graph_translator: Option<&crate::GraphTranslator>,
    ) -> crate::Result<()> {
        Ok(())
    }
}

impl AddGradFn for Base {
    #[inline]
    fn add_grad_fn<Args: Parents<N> + crate::UpdateArgs, const N: usize>(
        &self,
        _args: Args,
        _op: fn(&mut Args) -> crate::Result<()>,
    ) {
    }
}

#[cfg(feature = "autograd")]
impl crate::TapeActions for Base {}
