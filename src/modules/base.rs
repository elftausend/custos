use crate::{
    flag::AllocFlag, AddGradFn, AddOperation, Alloc, CachedBuffers, Cursor, Device, ExecNow, HasId,
    HashLocation, Module, OnDropBuffer, OnNewBuffer, Parents, PtrType, ReplaceBuf, Retrieve,
    SetOpHint, Setup, Shape, WrappedData,
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

    #[inline]
    fn add_op<Args: Parents<N>, const N: usize>(
        &self,
        mut args: Args,
        operation: fn(&mut Args) -> crate::Result<()>,
    ) -> crate::Result<()> {
        operation(&mut args)
    }

    #[inline]
    fn set_lazy_enabled(&self, _enabled: bool) {}

    #[inline]
    fn is_lazy_enabled(&self) -> bool {
        false
    }
}

impl<T> SetOpHint<T> for Base {}

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
    ) -> crate::Result<Self::Wrap<T, D::Base<T, S>>>
    where
        D: Alloc<T>,
    {
        device.alloc(len, AllocFlag::None)
    }
}

impl Cursor for Base {
    #[inline]
    fn cursor(&self) -> usize {
        0
    }

    #[inline]
    unsafe fn set_cursor(&self, _cursor: usize) {}
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

    #[inline]
    fn set_fork_enabled(&self, _enabled: bool) {}

    #[inline]
    fn is_fork_enabled(&self) -> bool {
        false
    }
}

#[cfg(feature = "graph")]
impl crate::Optimize for Base {
    #[inline]
    fn optimize_mem_graph<D: 'static>(
        &self,
        _device: &D,
        _graph_translator: Option<&crate::GraphTranslator>,
    ) -> crate::Result<()> {
        Ok(())
    }

    #[inline]
    fn unary_fusing<D: 'static>(
        &self,
        _device: &D,
        _graph_translator: Option<&crate::modules::GraphTranslator>,
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

    fn set_grad_enabled(&self, _enabled: bool) {}
}

#[cfg(feature = "autograd")]
impl crate::TapeActions for Base {}

impl CachedBuffers for Base {}
impl<T, D: Device, S: Shape> ReplaceBuf<T, D, S> for Base {
    #[inline]
    fn replace_buf<'a, 'c>(
        &'c self,
        buffer: &'c crate::Buffer<'a, T, D, S>,
    ) -> &'c crate::Buffer<'a, T, D, S> {
        buffer
    }
}
