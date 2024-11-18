use crate::{
    flag::AllocFlag, AddGradFn, AddOperation, Alloc, CachedBuffers, Cursor, Device, ExecNow, HasId,
    HashLocation, Module, OnNewBuffer, Parents, PtrType, ReplaceBuf, Retrieve,
    SetOpHint, Setup, Shape, Unit, WrappedData,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Base;

impl WrappedData for Base {
    type Wrap<'a, T: Unit, Base: 'static + HasId + PtrType> = Base;

    #[inline]
    fn wrap_in_base<'a, T: Unit, Base: 'static + HasId + PtrType>(
        &'a self,
        base: Base,
    ) -> Self::Wrap<'a, T, Base> {
        base
    }

    #[inline]
    fn wrapped_as_base<'a, 'b, T: Unit, Base: 'static + HasId + PtrType>(
        wrap: &'b Self::Wrap<'a, T, Base>,
    ) -> &'b Base {
        wrap
    }

    #[inline]
    fn wrapped_as_base_mut<'a, 'b, T: Unit, Base: 'static + HasId + PtrType>(
        wrap: &'b mut Self::Wrap<'a, T, Base>,
    ) -> &'b mut Base {
        wrap
    }

    #[inline]
    fn wrap_in_base_unbound<'a, T: Unit, Base: crate::IsBasePtr>(
        &self,
        base: Base,
    ) -> Self::Wrap<'a, T, Base> {
        base
    }
}

impl<'a, D: Device + 'a> Module<'a, D> for Base {
    type Module = Base;

    #[inline]
    fn new() -> Self::Module {
        Base
    }
}

impl AddOperation for Base {
    #[inline]
    fn add_op<Args: Parents<N> + crate::AnyOp, const N: usize>(
        &self,
        args: Args,
        op: impl for<'b> Fn(Args::Replicated<'b>) -> crate::Result<()> + 'static,
    ) -> crate::Result<()> {
        op(unsafe { args.replication() })
    }

    #[inline]
    fn ops_count(&self) -> usize {
        0
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

impl<'a, T: Unit, D: Device, S: Shape> OnNewBuffer<'a, T, D, S> for Base {}

impl<'a, D, T: Unit, S: Shape> Retrieve<'a, D, T, S> for Base {
    #[inline]
    fn retrieve_entry<const NUM_PARENTS: usize>(
        &'a self,
        device: &D,
        len: usize,
        parents: &impl Parents<NUM_PARENTS>,
    ) -> crate::Result<Self::Wrap<'a, T, D::Base<T, S>>>
    where
        D: Alloc<T>,
    {
        self.retrieve(device, len, parents)
    }

    #[inline]
    fn retrieve<const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        _parents: &impl Parents<NUM_PARENTS>,
    ) -> crate::Result<Self::Wrap<'a, T, <D>::Base<T, S>>>
    where
        S: Shape,
        D: Device + Alloc<T>,
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
    fn add_grad_fn<Args: Parents<N> + crate::AnyOp, const N: usize>(
        &self,
        _args: Args,
        _op: impl for<'b> Fn(Args::Replicated<'b>) -> crate::Result<()> + 'static,
    ) {
    }

    fn set_grad_enabled(&self, _enabled: bool) {}
}

#[cfg(feature = "autograd")]
impl crate::GradActions for Base {
    unsafe fn grad<
        'a,
        T: 'static,
        D: Device + Alloc<T> + crate::ZeroGrad<T> + 'static,
        S: Shape,
    >(
        &self,
        _device: &'a D,
        _buf: &crate::Buffer<'a, T, D, S>,
    ) -> &crate::Buffer<'a, T, D, S> {
        unimplemented!()
    }

    unsafe fn grad_mut<
        'a,
        T: 'static,
        D: Device + Alloc<T> + crate::ZeroGrad<T> + 'static,
        S: Shape,
    >(
        &self,
        _device: &'a D,
        _buf: &crate::Buffer<'a, T, D, S>,
    ) -> &mut crate::Buffer<'a, T, D, S> {
        unimplemented!()
    }
}

#[cfg(feature = "autograd")]
impl<'a> crate::TapeActions<'a> for Base {}

impl CachedBuffers for Base {}
impl<T: Unit, D: Device, S: Shape> ReplaceBuf<T, D, S> for Base {
    #[inline]
    fn replace_buf<'a, 'c>(
        &'c self,
        buffer: &'c crate::Buffer<'a, T, D, S>,
    ) -> &'c crate::Buffer<'a, T, D, S> {
        buffer
    }
}
