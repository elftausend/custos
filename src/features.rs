//! This module contains traits that are used to provide functionality to the modules.
//! Different modules can implement these traits to provide different functionality.
//! If the module does not need to alter the functionality, pass downs macros should be used to pass down the functionality to the wrapped module.

use core::{
    cell::{Ref, RefMut},
    fmt::Debug,
    ops::RangeBounds,
};

use crate::{
    AnyOp, CPU, HasId, Parents, Shape, UniqueId, Unit, WrappedData, ZeroGrad,
    op_hint::OpHint,
    range::{AsRange, CursorRange},
};

#[cfg(feature = "cached")]
use crate::{Base, CachedModule};

use super::{Alloc, Buffer, Device};

pub trait Feature: WrappedData {}

// is a cached module is placed before Autograd results a problem
// -> the retrieved buffer is not added to the no grads pool of the autograd module
// let device = CPU::<Cached<Autograd<Base>>>::new();
//
// how to fix this:
// add retrieved buffer to no grads pool at the end of the chain (at device level (Retriever trait))
// => "generator", "actor"
pub trait Retrieve<D, T: Unit, S: Shape = ()>: WrappedData {
    // "generator"
    #[track_caller]
    fn retrieve_entry<'a, const NUM_PARENTS: usize>(
        &'a self,
        device: &D,
        len: usize,
        parents: &impl Parents<NUM_PARENTS>,
    ) -> crate::Result<Self::Wrap<'a, T, D::Base<T, S>>>
    where
        S: Shape,
        D: Alloc<T>;

    #[track_caller]
    fn retrieve<'a, const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        parents: &impl Parents<NUM_PARENTS>,
    ) -> crate::Result<Self::Wrap<'a, T, D::Base<T, S>>>
    where
        S: Shape,
        D: Alloc<T>;

    // "actor"
    #[inline]
    fn on_retrieve_finish<const NUM_PARENTS: usize>(
        &self,
        _len: usize,
        _parents: impl Parents<NUM_PARENTS>,
        _retrieved_buf: &Buffer<T, D, S>,
    ) where
        D: Alloc<T>,
    {
    }
}

pub trait Cursor {
    fn cursor(&self) -> usize;

    /// # Safety
    /// Moving the cursor manually to a specific position can possible create multiple mutable references to the same memory location
    unsafe fn set_cursor(&self, cursor: usize);

    /// # Safety
    /// Moving the cursor manually to a specific position can possible create multiple mutable references to the same memory location
    #[inline]
    unsafe fn inc_cursor(&self, inc: usize) {
        unsafe { self.set_cursor(self.cursor() + inc) }
    }

    /// # Safety
    /// Moving the cursor manually to a specific position can possible create multiple mutable references to the same memory location
    #[inline]
    unsafe fn bump_cursor(&self) {
        unsafe { self.inc_cursor(1) }
    }

    #[inline]
    fn range(&self, range: impl AsRange) -> CursorRange<Self>
    where
        Self: Sized,
    {
        CursorRange {
            start: range.start(),
            end: range.end(),
            device: self,
        }
    }

    #[inline]
    fn cached(&self, cb: impl Fn())
    where
        Self: Sized,
    {
        let mut range = self.range(1).into_iter();
        cb();

        range.next();
    }
}

#[macro_export]
macro_rules! pass_down_cursor {
    ($to_impl:ident, $($generics:tt),*) => {
        impl<'dev, Mods: $crate::Cursor> $crate::Cursor for $to_impl<$($generics),*> {
            #[inline]
            fn cursor(&self) -> usize {
                self.modules.cursor()
            }

            #[inline]
            unsafe fn set_cursor(&self, cursor: usize) {
                unsafe { self.modules.set_cursor(cursor) }
            }
        }
    };
    ($to_impl:ident) => {
        $crate::pass_down_cursor!($to_impl, Mods);
    };
}

/// Used for modules that should affect the device.
pub trait Setup<D> {
    #[inline]
    fn setup(_device: &mut D) -> crate::Result<()> {
        Ok(())
    }
}

// only for base and lazy?
pub trait RunModule<D> {
    #[inline]
    fn run(&self, _device: &D) -> crate::Result<()> {
        Ok(())
    }
}

impl<D> RunModule<D> for crate::Base {}

pub trait Run {
    /// Executes a lazy graph.
    fn run(&self) -> crate::Result<()>;
}

pub trait HasModules {
    type Mods;
    fn modules(&self) -> &Self::Mods;
}

#[cfg(feature = "autograd")]
pub trait GradActions {
    #[inline]
    unsafe fn gradients(&self) -> Option<&crate::Gradients> {
        None
    }

    #[inline]
    unsafe fn gradients_mut(&self) -> Option<&mut crate::Gradients> {
        None
    }
    unsafe fn grad<'a, T: 'static, D: Device + Alloc<T> + ZeroGrad<T> + 'static, S: Shape>(
        &self,
        device: &'a D,
        buf: &Buffer<'a, T, D, S>,
    ) -> &Buffer<'static, T, D, S>;

    #[allow(clippy::mut_from_ref)]
    unsafe fn grad_mut<'a, T: 'static, D: Device + Alloc<T> + ZeroGrad<T> + 'static, S: Shape>(
        &self,
        device: &'a D,
        buf: &Buffer<'a, T, D, S>,
    ) -> &mut Buffer<'static, T, D, S>;
}

pub trait AddGradFn {
    fn add_grad_fn_inner<D: 'static, Args: Parents<N> + AnyOp, const N: usize>(
        &self,
        args: Args,
        device: &D,
        op: impl for<'b> Fn(Args::Replicated<'b>, &D) -> crate::Result<()> + 'static,
    );

    fn add_grad_fn<Args: Parents<N> + AnyOp, const N: usize>(
        &self,
        args: Args,
        op: impl for<'b> Fn(Args::Replicated<'b>, &Self) -> crate::Result<()> + 'static,
    ) where Self: Device + 'static;
    
    fn add_grad_and_forward_fn<
        D: Device + 'static,
        Args: Parents<N> + AnyOp + Clone,
        const N: usize,
    >(
        &self,
        args: Args,
        device: &D,
        forward_fn: impl for<'b> Fn(Args::Replicated<'b>, &D) -> crate::Result<()> + 'static,
        grad_fn: impl for<'b> Fn(Args::Replicated<'b>, &D) -> crate::Result<()> + 'static,
    ) where
        Self: AddOperation,
    {
        self.add_op_inner(args.clone(), device, forward_fn).unwrap();
        self.add_grad_fn_inner(args, device, grad_fn)
    }

    fn backward(&mut self) {}

    #[inline]
    fn disable_grad(&self) {
        self.set_grad_enabled(false)
    }

    #[inline]
    fn enable_grad(&self) {
        self.set_grad_enabled(true)
    }

    fn set_grad_enabled(&self, enabled: bool);
    #[inline]
    fn is_grad_enabled(&self) -> bool {
        false
    }

    /// This disables gradient function tracking, calls `op`, then reverts to the previous tracking mode
    /// # Example
    #[cfg_attr(all(feature = "cpu", feature = "autograd"), doc = "```")]
    #[cfg_attr(not(all(feature = "cpu", feature = "autograd")), doc = "```ignore")]
    /// use custos::prelude::*;
    ///
    /// let device = CPU::<Autograd<Base>>::new();
    ///
    /// let lhs = device.buffer([1, 2, 3, 4, 5]);
    /// device.no_grad_ctx(|| {
    ///     device.add_grad_fn(&lhs, &device, |lhs, _| {
    ///         panic!("should not execute!");
    ///         Ok(())
    ///     })
    /// });
    ///
    /// lhs.backward();
    /// ```
    #[inline]
    fn no_grad_ctx(&self, op: impl FnOnce()) {
        let enabled_before = self.is_grad_enabled();
        self.set_grad_enabled(false);
        op();
        self.set_grad_enabled(enabled_before);
    }
}

#[macro_export]
macro_rules! pass_down_grad_fn {
    ($to_impl:ident, $($generics:tt),*) => {
        #[cfg(feature = "autograd")]
        impl<'dev, Mods: $crate::GradActions> $crate::GradActions for $to_impl<$($generics),*> {
            unsafe fn grad<'a, T: 'static, D: Device + $crate::Alloc<T> + $crate::ZeroGrad<T> + 'static, S: Shape>(
                &self,
                device: &'a D,
                buf: &Buffer<'a, T, D, S>,
            ) -> &Buffer<'static, T, D, S> {
                unsafe { self.modules.grad(device, buf) }
            }

            unsafe fn grad_mut<'a, T: 'static, D: Device + $crate::Alloc<T> + $crate::ZeroGrad<T> + 'static, S: Shape>(
                &self,
                device: &'a D,
                buf: &Buffer<'a, T, D, S>,
            ) -> &mut Buffer<'static, T, D, S> {
                unsafe { self.modules.grad_mut(device, buf) }
            }

            #[inline]
            unsafe fn gradients(&self) -> Option<&$crate::Gradients> {
                unsafe { self.modules.gradients() }
            }

            #[inline]
            unsafe fn gradients_mut(&self) -> Option<&mut $crate::Gradients> {
                unsafe { self.modules.gradients_mut() }
            }

        }
        impl<'dev, Mods: $crate::AddGradFn> $crate::AddGradFn for $to_impl<$($generics),*> {
            #[inline]
            fn add_grad_fn_inner<D: 'static, Args: $crate::Parents<N> + $crate::AnyOp, const N: usize>(
                &self,
                args: Args,
                device: &D,
                op: impl for<'b> Fn(Args::Replicated<'b>, &D) -> $crate::Result<()> + 'static,
            ) {
                self.modules.add_grad_fn_inner(args, device, op);
            }
            
            #[inline]
            fn add_grad_fn<Args: $crate::Parents<N> + $crate::AnyOp, const N: usize>(
                &self,
                args: Args,
                op: impl for<'b> Fn(Args::Replicated<'b>, &Self) -> $crate::Result<()> + 'static,
            ) where Self: $crate::Device + 'static {
                self.modules.add_grad_fn_inner(args, self, op);
            }

            #[inline]
            fn backward(&mut self) {
                self.modules.backward()
            }

            #[inline]
            fn set_grad_enabled(&self, enabled: bool) {
                self.modules.set_grad_enabled(enabled)
            }
        }
    };
    ($to_impl:ident) => {
        $crate::pass_down_grad_fn!($to_impl, Mods);
    };
}

#[cfg(feature = "autograd")]
pub trait TapeActions<'dev> {
    // "generator" - do not forget to pass down
    #[inline]
    fn tape(&self) -> Option<Ref<crate::Tape<'dev>>> {
        None
    }
    // "generator" - do not forget to pass down
    #[inline]
    fn tape_mut(&self) -> Option<RefMut<crate::Tape<'dev>>> {
        None
    }
}

#[macro_export]
macro_rules! pass_down_tape_actions {
    ($to_impl:ident, $($generics:tt),*) => {
        #[cfg(feature = "autograd")]
        impl<'dev, Mods: $crate::HasAutograd> $crate::HasAutograd for $to_impl<$($generics),*> {}


        #[cfg(feature = "autograd")]
        impl<'dev, Mods: $crate::TapeActions<'dev>> $crate::TapeActions<'dev> for $to_impl<$($generics),*>
        where
            Self: 'dev
        {
            #[inline]
            fn tape(&self) -> Option<core::cell::Ref<$crate::Tape<'dev>>> {
                self.modules.tape()
            }

            #[inline]
            fn tape_mut(&self) -> Option<core::cell::RefMut<$crate::Tape<'dev>>> {
                self.modules.tape_mut()
            }
        }
    };
    ($to_impl:ident) => {
        $crate::pass_down_tape_actions!($to_impl, Mods);
    };
}

pub trait OpArgs {
    fn as_ids(&self) -> [UniqueId; 2];
}

impl<'a, 'b, T: Unit, D: Device, S: Shape> OpArgs for (&Buffer<'a, T, D, S>, &Buffer<'b, T, D, S>) {
    fn as_ids(&self) -> [UniqueId; 2] {
        [*self.0.id(), *self.1.id()]
    }
}

// seems useless, however, this is used to retrieve potential lazy buffer information
pub trait ReplaceBuf<T: Unit, D: Device, S: Shape>: WrappedData {
    fn replace_buf<'a, 'c>(&'c self, buffer: &'c Buffer<'a, T, D, S>) -> &'c Buffer<'a, T, D, S>;
    fn set_checkpoint_buffer(&self, buffer_id: &crate::Id);
}

pub trait ReplaceBufPassDown {}

impl<'b, T, S, D, Mod> ReplaceBuf<T, D, S> for Mod
where
    <Mod as HasModules>::Mods: ReplaceBuf<T, D, S>,
    T: Unit,
    S: Shape,
    D: Device + 'b,
    Mod: ReplaceBufPassDown + HasModules + WrappedData,
{
    #[inline]
    fn replace_buf<'a, 'c>(&'c self, buffer: &'c Buffer<'a, T, D, S>) -> &'c Buffer<'a, T, D, S> {
        self.modules().replace_buf(buffer)
    }

    fn set_checkpoint_buffer(&self, buffer_id: &crate::Id) {
        self.modules().set_checkpoint_buffer(buffer_id);
    }
}

pub trait AddOperation {
    fn add_op_inner<D: Device + 'static, Args: Parents<N> + AnyOp, const N: usize>(
        &self,
        args: Args,
        device: &D,
        op: impl for<'b> Fn(Args::Replicated<'b>, &D) -> crate::Result<()> + 'static,
    ) -> crate::Result<()>;

    fn add_op<Args: Parents<N> + AnyOp, const N: usize>(
        &self,
        args: Args,
        op: impl for<'b> Fn(Args::Replicated<'b>, &Self) -> crate::Result<()> + 'static,
    ) -> crate::Result<()>
    where
        Self: Device + 'static;

    fn ops_count(&self) -> usize;
    fn set_lazy_enabled(&self, enabled: bool);
    #[inline]
    fn enable_lazy(&self) {
        self.set_lazy_enabled(true)
    }
    #[inline]
    fn disable_lazy(&self) {
        self.set_lazy_enabled(false)
    }
    fn is_lazy_enabled(&self) -> bool;

    #[inline]
    fn eagerly(&self, op: impl FnOnce()) {
        // use enabled_before -> if eager is called in an already disabled lazy context, it should not be enabled after the call
        let enabled_before = self.is_lazy_enabled();
        self.set_lazy_enabled(false);
        op();
        self.set_lazy_enabled(enabled_before);
    }
}

pub trait AddOperationPassDown {}

impl<'b, Module> AddOperation for Module
where
    <Module as HasModules>::Mods: AddOperation,
    Module: AddOperationPassDown + HasModules + WrappedData,
{
    fn add_op_inner<D: Device + 'static, Args: Parents<N> + AnyOp, const N: usize>(
        &self,
        args: Args,
        device: &D,
        op: impl for<'a> Fn(Args::Replicated<'a>, &D) -> crate::Result<()> + 'static,
    ) -> crate::Result<()> {
        self.modules().add_op_inner(args, device, op)
    }

    fn add_op<Args: Parents<N> + AnyOp, const N: usize>(
        &self,
        args: Args,
        op: impl for<'a> Fn(Args::Replicated<'a>, &Self) -> crate::Result<()> + 'static,
    ) -> crate::Result<()>
    where
        Self: Device + 'static,
    {
        self.modules().add_op_inner(args, self, op)
    }

    #[inline]
    fn ops_count(&self) -> usize {
        self.modules().ops_count()
    }

    #[inline]
    fn set_lazy_enabled(&self, enabled: bool) {
        self.modules().set_lazy_enabled(enabled)
    }

    #[inline]
    fn is_lazy_enabled(&self) -> bool {
        self.modules().is_lazy_enabled()
    }
}

pub trait SetOpHint<T> {
    fn set_op_hint(&self, _op_hint: OpHint<T>) {}
}

impl<'b, Module, T> SetOpHint<T> for Module
where
    <Module as HasModules>::Mods: SetOpHint<T>,
    Module: AddOperationPassDown + HasModules + WrappedData,
{
    #[inline]
    fn set_op_hint(&self, op_hint: OpHint<T>) {
        self.modules().set_op_hint(op_hint);
    }
}

pub trait ExecNow<D = Self> {
    /// This drains the affected operations!
    fn exec_now(&self, device: &D, range_bounds: impl RangeBounds<usize>) -> crate::Result<()>;

    /// This drains the affected operations!
    #[inline]
    fn exec_last_n(&self, device: &D, last_n: usize) -> crate::Result<()>
    where
        D: Device,
        Self: AddOperation,
    {
        self.exec_now(device, self.ops_count() - last_n..)
    }
}

pub trait ExecNowPassDown {}

impl<'b, D, Module> ExecNow<D> for Module
where
    <Module as HasModules>::Mods: ExecNow<D>,
    D: Device + 'b,
    Module: ExecNowPassDown + HasModules + WrappedData,
{
    #[inline]
    fn exec_now(&self, device: &D, range_bounds: impl RangeBounds<usize>) -> crate::Result<()> {
        self.modules().exec_now(device, range_bounds)
    }
}

pub trait HasCPU<Mods> {
    fn cpu(&self) -> &CPU<Mods>;
}

#[cfg(feature = "cached")]
pub type CachedCPU = CPU<CachedModule<Base, CPU>>;

#[cfg(feature = "cached")]
pub trait UnifiedMemChain<D: Device> {
    fn construct_unified_buf_from_cpu_buf<'a, T: Unit + 'static, S: Shape>(
        &self,
        device: &'a D,
        no_drop_buf: Buffer<'a, T, CachedCPU, S>,
    ) -> crate::Result<Buffer<'a, T, D, S>>;
}

#[cfg(feature = "cached")]
#[macro_export]
macro_rules! pass_down_unified_mem_chain {
    ($to_impl:ident, $($generics:tt),*) => {
        impl<'dev, Mods: $crate::UnifiedMemChain<D>, D: Device> $crate::UnifiedMemChain<D> for $to_impl<$($generics),*> {
            fn construct_unified_buf_from_cpu_buf<'a, T: $crate::Unit + 'static, S: Shape>(
                &self,
                device: &'a D,
                no_drop_buf: Buffer<'a, T, $crate::CachedCPU, S>
            ) -> $crate::Result<Buffer<'a, T, D, S>>
            {
                self.modules.construct_unified_buf_from_cpu_buf(device, no_drop_buf)
            }
        }
    };

    ($to_impl:ident) => {
        $crate::pass_down_unified_mem_chain!($to_impl, Mods);
    };
}

#[cfg(feature = "autograd")]
use crate::Autograd;
#[cfg(feature = "lazy")]
use crate::Lazy;

#[cfg(feature = "lazy")]
#[cfg(feature = "cached")]
pass_down_unified_mem_chain!(Lazy, 'dev, Mods);

#[cfg(feature = "autograd")]
pass_down_unified_mem_chain!(Autograd, 'dev, Mods);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct GpuOrCpuInfo {
    pub use_cpu: bool,
    pub is_result_cached: bool,
}

#[macro_export]
macro_rules! pass_down_use_gpu_or_cpu {
    ($to_impl:ident, $($generics:tt),*) => {
        impl<'dev, Mods: $crate::UseGpuOrCpu> $crate::UseGpuOrCpu for $to_impl<$($generics),*> {
            #[inline]
            fn use_cpu_or_gpu(
                &self,
                location: $crate::HashLocation<'static>,
                input_lengths: &[usize],
                cpu_op: impl FnMut(),
                gpu_op: impl FnMut(),
            ) -> $crate::GpuOrCpuInfo {
                self.modules
                    .use_cpu_or_gpu(location, input_lengths, cpu_op, gpu_op)
            }

            #[inline]
            fn set_fork_enabled(&self, enabled: bool) {
                self.modules.set_fork_enabled(enabled);
            }

            fn is_fork_enabled(&self) -> bool {
                self.modules.is_fork_enabled()
            }
        }
    };
    ($to_impl:ident) => {
        $crate::pass_down_use_gpu_or_cpu!($to_impl, Mods);
    };
}

#[cfg(feature = "autograd")]
pass_down_use_gpu_or_cpu!(Autograd, 'dev, Mods);

pub trait UseGpuOrCpu {
    fn use_cpu_or_gpu(
        &self,
        location: crate::HashLocation<'static>,
        input_lengths: &[usize],
        cpu_op: impl FnMut(),
        gpu_op: impl FnMut(),
    ) -> GpuOrCpuInfo;

    #[track_caller]
    #[inline]
    fn use_cpu_or_gpu_tracked(
        &self,
        input_lengths: &[usize],
        cpu_op: impl FnMut(),
        gpu_op: impl FnMut(),
    ) -> GpuOrCpuInfo {
        let location = crate::HashLocation::from(core::panic::Location::caller());
        self.use_cpu_or_gpu(location, input_lengths, cpu_op, gpu_op)
    }

    fn set_fork_enabled(&self, _enabled: bool);

    #[inline]
    fn disable_fork(&self) {
        self.set_fork_enabled(false)
    }

    #[inline]
    fn enable_fork(&self) {
        self.set_fork_enabled(true)
    }

    fn is_fork_enabled(&self) -> bool;
}

#[cfg(feature = "graph")]
pub trait Optimize {
    unsafe fn optimize_mem_graph<D: 'static>(
        &self,
        device: &D,
        graph_translator: Option<&crate::modules::GraphTranslator>,
    ) -> crate::Result<()>;

    fn unary_fusing<D: crate::UnaryFusing + 'static>(
        &self,
        device: &D,
        graph_translator: Option<&crate::modules::GraphTranslator>,
    ) -> crate::Result<()>;
}

#[macro_export]
macro_rules! pass_down_optimize_mem_graph {
    ($to_impl:ident, $($generics:tt),*) => {
        impl<'dev, Mods: $crate::Optimize> $crate::Optimize for $to_impl<$($generics),*> {
            unsafe fn optimize_mem_graph<D: 'static>(
                &self,
                device: &D,
                graph_translator: Option<&$crate::modules::GraphTranslator>,
            ) -> $crate::Result<()> {
                unsafe { self.modules.optimize_mem_graph(device, graph_translator) }
            }
            fn unary_fusing<D: $crate::UnaryFusing + 'static>(
                &self,
                device: &D,
                graph_translator: Option<&$crate::modules::GraphTranslator>,
            ) -> $crate::Result<()> {
                self.modules.unary_fusing(device, graph_translator)
            }
        }
    };
    ($to_impl:ident) => {
        $crate::pass_down_optimize_mem_graph!($to_impl, Mods);
    };
}

pub trait CachedBuffers {
    #[cfg(feature = "std")]
    unsafe fn buffers_mut(
        &self,
    ) -> Option<RefMut<crate::Buffers<Box<dyn crate::BoxedShallowCopy>>>> {
        None
    }

    #[inline]
    fn are_cached_buffers_supplied_from_below_module(&self) -> bool {
        false
    }
}

#[macro_export]
macro_rules! pass_down_cached_buffers {
    ($to_impl:ident, $($generics:tt),*) => {
        impl<'dev, Mods: $crate::CachedBuffers> $crate::CachedBuffers for $to_impl<$($generics),*> {
            #[cfg(feature = "std")]
            #[inline]
            unsafe fn buffers_mut(
                &self,
            ) -> Option<core::cell::RefMut<$crate::Buffers<Box<dyn $crate::BoxedShallowCopy>>>>
            {
                unsafe { self.modules.buffers_mut() }
            }
        }
    };
    ($to_impl:ident) => {
        $crate::pass_down_cached_buffers!($to_impl, Mods);
    };
}
