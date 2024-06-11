//! This module contains traits that are used to provide functionality to the modules.
//! Different modules can implement these traits to provide different functionality.
//! If the module does not need to alter the functionality, pass downs macros should be used to pass down the functionality to the wrapped module.

use core::{cell::RefMut, fmt::Debug, ops::RangeBounds};

use crate::{
    op_hint::OpHint,
    range::{AsRange, CursorRange},
    HasId, Parents, Shape, UniqueId, Unit, UpdateArgs, CPU,
};

#[cfg(feature = "cached")]
use crate::{Base, CachedModule};

use super::{Alloc, Buffer, Device, OnDropBuffer};

pub trait Feature: OnDropBuffer {}

// is a cached module is placed before Autograd results a problem
// -> the retrieved buffer is not added to the no grads pool of the autograd module
// let device = CPU::<Cached<Autograd<Base>>>::new();
//
// how to fix this:
// add retrieved buffer to no grads pool at the end of the chain (at device level (Retriever trait))
// => "generator", "actor"
pub trait Retrieve<D, T: Unit, S: Shape = ()>: OnDropBuffer {
    // "generator"
    #[track_caller]
    unsafe fn retrieve<const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        parents: impl Parents<NUM_PARENTS>,
    ) -> crate::Result<Self::Wrap<T, D::Base<T, S>>>
    where
        S: Shape,
        D: Device + Alloc<T>;

    // "actor"
    #[inline]
    fn on_retrieve_finish(&self, _retrieved_buf: &Buffer<T, D, S>)
    where
        D: Alloc<T>,
    {
    }
}

pub trait Cursor {
    fn cursor(&self) -> usize;
    unsafe fn set_cursor(&self, cursor: usize);

    #[inline]
    unsafe fn inc_cursor(&self, inc: usize) {
        self.set_cursor(self.cursor() + inc)
    }

    #[inline]
    unsafe fn bump_cursor(&self) {
        self.inc_cursor(1)
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
}

#[macro_export]
macro_rules! pass_down_cursor {
    ($to_impl:ident) => {
        impl<Mods: $crate::Cursor> $crate::Cursor for $to_impl<Mods> {
            #[inline]
            fn cursor(&self) -> usize {
                self.modules.cursor()
            }

            #[inline]
            unsafe fn set_cursor(&self, cursor: usize) {
                self.modules.set_cursor(cursor)
            }
        }
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
    ///
    /// # Safety
    /// The lifetime of captured references is ignored!
    /// Specific style of writing operations should prevent UB altogether (at the cost of convenience).
    unsafe fn run(&self) -> crate::Result<()>;
}

pub trait HasModules {
    type Mods;
    fn modules(&self) -> &Self::Mods;
}

pub trait AddGradFn {
    fn add_grad_fn<Args: Parents<N> + UpdateArgs, const N: usize>(
        &self,
        args: Args,
        op: fn(&mut Args) -> crate::Result<()>,
    );

    fn add_grad_and_forward_fn<Args: Parents<N> + UpdateArgs + Clone, const N: usize>(
        &self,
        args: Args,
        forward_fn: fn(&mut Args) -> crate::Result<()>,
        grad_fn: fn(&mut Args) -> crate::Result<()>,
    ) where
        Self: AddOperation,
    {
        self.add_op(args.clone(), forward_fn).unwrap();
        self.add_grad_fn(args, grad_fn)
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
    ///     device.add_grad_fn(&lhs, |lhs| {
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
    ($to_impl:ident) => {
        impl<Mods: $crate::AddGradFn> $crate::AddGradFn for $to_impl<Mods> {
            #[inline]
            fn add_grad_fn<Args: $crate::Parents<N> + $crate::UpdateArgs, const N: usize>(
                &self,
                args: Args,
                op: fn(&mut Args) -> $crate::Result<()>,
            ) {
                self.modules.add_grad_fn(args, op)
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
}

#[cfg(feature = "autograd")]
pub trait TapeActions {
    // "generator" - do not forget to pass down
    #[inline]
    unsafe fn tape(&self) -> Option<&crate::Tape> {
        None
    }
    // "generator" - do not forget to pass down
    #[inline]
    unsafe fn tape_mut(&self) -> Option<&mut crate::Tape> {
        None
    }

    #[inline]
    unsafe fn gradients(&self) -> Option<&crate::Gradients> {
        None
    }

    #[inline]
    unsafe fn gradients_mut(&self) -> Option<&mut crate::Gradients> {
        None
    }
}

#[macro_export]
macro_rules! pass_down_tape_actions {
    ($to_impl:ident) => {
        #[cfg(feature = "autograd")]
        impl<Mods: $crate::HasAutograd> $crate::HasAutograd for $to_impl<Mods> {}

        #[cfg(feature = "autograd")]
        impl<Mods: $crate::TapeActions> $crate::TapeActions for $to_impl<Mods> {
            #[inline]
            unsafe fn tape(&self) -> Option<&$crate::Tape> {
                self.modules.tape()
            }

            #[inline]
            unsafe fn tape_mut(&self) -> Option<&mut $crate::Tape> {
                self.modules.tape_mut()
            }

            #[inline]
            unsafe fn gradients(&self) -> Option<&$crate::Gradients> {
                self.modules.gradients()
            }

            #[inline]
            unsafe fn gradients_mut(&self) -> Option<&mut $crate::Gradients> {
                self.modules.gradients_mut()
            }
        }
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
pub trait ReplaceBuf<T: Unit, D: Device, S: Shape>: OnDropBuffer {
    fn replace_buf<'a, 'c>(&'c self, buffer: &'c Buffer<'a, T, D, S>) -> &'c Buffer<'a, T, D, S>;
}

#[macro_export]
macro_rules! pass_down_replace_buf_dev {
    ($device:ident) => {
        impl<T: $crate::Unit, S: Shape, Mods: $crate::ReplaceBuf<T, Self, S>>
            $crate::ReplaceBuf<T, Self, S> for $device<Mods>
        {
            #[inline]
            fn replace_buf<'a, 'c>(
                &'c self,
                buffer: &'c Buffer<'a, T, Self, S>,
            ) -> &'c Buffer<'a, T, Self, S> {
                self.modules.replace_buf(buffer)
            }
        }
    };
}

#[macro_export]
macro_rules! pass_down_replace_buf_module {
    ($module:ident) => {
        impl<T: $crate::Unit, S: Shape, Mods: $crate::ReplaceBuf<T, D, S>, D: $crate::Device>
            $crate::ReplaceBuf<T, D, S> for $module<Mods>
        {
            #[inline]
            fn replace_buf<'a, 'c>(
                &'c self,
                buffer: &'c Buffer<'a, T, D, S>,
            ) -> &'c Buffer<'a, T, D, S> {
                self.modules.replace_buf(buffer)
            }
        }
    };
}

pub trait AddOperation {
    fn add_op<Args: Parents<N> + UpdateArgs, const N: usize>(
        &self,
        args: Args,
        operation: fn(&mut Args) -> crate::Result<()>,
    ) -> crate::Result<()>;
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

pub trait SetOpHint<T> {
    fn set_op_hint(&self, _op_hint: OpHint<T>) {}
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

/// Implements the [`AddOperation`] trait for any supplied device. The `add_op` call is passed down to `self.modules`.
#[macro_export]
macro_rules! pass_down_add_operation {
    ($device:ident) => {
        impl<T, Mods: $crate::SetOpHint<T>> $crate::SetOpHint<T> for $device<Mods> {
            #[inline]
            fn set_op_hint(&self, op_hint: $crate::op_hint::OpHint<T>) {
                self.modules.set_op_hint(op_hint)
            }
        }

        impl<Mods: $crate::AddOperation> $crate::AddOperation for $device<Mods> {
            #[inline]
            fn add_op<Args: $crate::Parents<N> + $crate::UpdateArgs, const N: usize>(
                &self,
                args: Args,
                operation: fn(&mut Args) -> $crate::Result<()>,
            ) -> $crate::Result<()> {
                self.modules.add_op(args, operation)
            }

            #[inline]
            fn ops_count(&self) -> usize {
                self.modules.ops_count()
            }

            #[inline]
            fn set_lazy_enabled(&self, enabled: bool) {
                self.modules.set_lazy_enabled(enabled)
            }

            #[inline]
            fn is_lazy_enabled(&self) -> bool {
                self.modules.is_lazy_enabled()
            }
        }
    };
}

#[macro_export]
macro_rules! pass_down_exec_now_module {
    ($device:ident) => {
        impl<D: $crate::Device, Mods: $crate::ExecNow<D>> $crate::ExecNow<D> for $device<Mods> {
            #[inline]
            fn exec_now(
                &self,
                device: &D,
                range_bounds: impl core::ops::RangeBounds<usize>,
            ) -> $crate::Result<()> {
                self.modules.exec_now(device, range_bounds)
            }
        }
    };
}

// FIXME may remove for device and another trait for devices (mind device ref in exec noe)
#[macro_export]
macro_rules! pass_down_exec_now {
    ($device:ident) => {
        impl<Mods: $crate::ExecNow<Self>> $crate::ExecNow<Self> for $device<Mods> {
            #[inline]
            fn exec_now(
                &self,
                device: &Self,
                range_bounds: impl core::ops::RangeBounds<usize>,
            ) -> $crate::Result<()> {
                self.modules.exec_now(device, range_bounds)
            }
        }
    };
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
    ($($to_impl:ident),*) => {
        $(
            impl<Mods: $crate::UnifiedMemChain<D>, D: Device> $crate::UnifiedMemChain<D> for $to_impl<Mods> {
                fn construct_unified_buf_from_cpu_buf<'a, T: $crate::Unit + 'static, S: Shape>(
                    &self,
                    device: &'a D,
                    no_drop_buf: Buffer<'a, T, $crate::CachedCPU, S>
                ) -> $crate::Result<Buffer<'a, T, D, S>>
                {
                    self.modules.construct_unified_buf_from_cpu_buf(device, no_drop_buf)
                }
            }

        )*
    };
}

#[cfg(feature = "autograd")]
use crate::Autograd;
#[cfg(feature = "lazy")]
use crate::Lazy;

#[cfg(feature = "lazy")]
#[cfg(feature = "cached")]
pass_down_unified_mem_chain!(Lazy);

#[cfg(feature = "autograd")]
pass_down_unified_mem_chain!(Autograd);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct GpuOrCpuInfo {
    pub use_cpu: bool,
    pub is_result_cached: bool,
}

#[macro_export]
macro_rules! pass_down_use_gpu_or_cpu {
    ($to_impl:ident) => {
        impl<Mods: $crate::UseGpuOrCpu> $crate::UseGpuOrCpu for $to_impl<Mods> {
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
}

#[cfg(feature = "autograd")]
pass_down_use_gpu_or_cpu!(Autograd);

pub trait UseGpuOrCpu {
    fn use_cpu_or_gpu(
        &self,
        location: crate::HashLocation<'static>,
        input_lengths: &[usize],
        cpu_op: impl FnMut(),
        gpu_op: impl FnMut(),
    ) -> GpuOrCpuInfo;

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
    fn optimize_mem_graph<D: 'static>(
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
    ($to_impl:ident) => {
        impl<Mods: $crate::Optimize> $crate::Optimize for $to_impl<Mods> {
            fn optimize_mem_graph<D: 'static>(
                &self,
                device: &D,
                graph_translator: Option<&$crate::modules::GraphTranslator>,
            ) -> $crate::Result<()> {
                self.modules.optimize_mem_graph(device, graph_translator)
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
}

pub trait CachedBuffers {
    #[cfg(feature = "std")]
    unsafe fn buffers_mut(
        &self,
    ) -> Option<RefMut<crate::Buffers<Box<dyn crate::BoxedShallowCopy>>>> {
        None
    }
}

#[macro_export]
macro_rules! pass_down_cached_buffers {
    ($to_impl:ident) => {
        impl<Mods: $crate::CachedBuffers> $crate::CachedBuffers for $to_impl<Mods> {
            #[cfg(feature = "std")]
            #[inline]
            unsafe fn buffers_mut(
                &self,
            ) -> Option<core::cell::RefMut<$crate::Buffers<Box<dyn $crate::BoxedShallowCopy>>>>
            {
                self.modules.buffers_mut()
            }
        }
    };
}
