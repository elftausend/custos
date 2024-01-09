use core::{fmt::Debug, ops::RangeBounds};

use crate::{HasId, Parents, Shape, UniqueId, UpdateArgs, CPU, Buffers};

#[cfg(feature = "graph")]
use crate::HashLocationCacheTrace;

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
pub trait Retrieve<D, T, S: Shape = ()>: OnDropBuffer {
    // "generator"
    #[track_caller]
    fn retrieve<const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        parents: impl Parents<NUM_PARENTS>,
    ) -> Self::Wrap<T, D::Base<T, S>>
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

pub trait HasModules<Mods> {
    fn modules(&self) -> &Mods;
}

pub trait AddGradFn {
    fn add_grad_fn<Args: Parents<N> + UpdateArgs<Buffers>, const N: usize>(
        &self,
        args: Args,
        op: fn(&mut Args) -> crate::Result<()>,
    );

    fn add_grad_and_forward_fn<Args: Parents<N> + UpdateArgs<Buffers> + Clone, const N: usize>(
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
}

#[macro_export]
macro_rules! pass_down_grad_fn {
    ($to_impl:ident) => {
        impl<Mods: $crate::AddGradFn> $crate::AddGradFn for $to_impl<Mods> {
            #[inline]
            fn add_grad_fn<Args: $crate::Parents<N> + $crate::UpdateArgs<$crate::Buffers>, const N: usize>(
                &self,
                args: Args,
                op: fn(&mut Args) -> crate::Result<()>,
            ) {
                self.modules.add_grad_fn(args, op)
            }

            #[inline]
            fn backward(&mut self) {
                self.modules.backward()
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
    // fn update_vals(&mut self, cache ..)
    // fn from_cache(cache: &std::collections::HashMap<UniqueId, ()>, ids: [UniqueId; N]) -> Self;
}

impl<'a, 'b, T, D: Device, S: Shape> OpArgs for (&Buffer<'a, T, D, S>, &Buffer<'b, T, D, S>) {
    fn as_ids(&self) -> [UniqueId; 2] {
        [*self.0.id(), *self.1.id()]
    }
}

// seems useless, however, this is used to retrieve potential lazy buffer information
pub trait ReplaceBuf<T, D: Device, S: Shape>: OnDropBuffer {
    fn replace_buf<'a, 'c>(&'c self, buffer: &'c Buffer<'a, T, D, S>) -> &'c Buffer<'a, T, D, S>;
}

#[macro_export]
macro_rules! pass_down_replace_buf {
    ($device:ident) => {
        impl<T, S: Shape, Mods: $crate::ReplaceBuf<T, Self, S>> $crate::ReplaceBuf<T, Self, S>
            for $device<Mods>
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

pub trait AddOperation {
    #[track_caller]
    fn add_op<Args: Parents<N> + UpdateArgs<Buffers>, const N: usize>(
        &self,
        args: Args,
        operation: fn(&mut Args) -> crate::Result<()>,
    ) -> crate::Result<()>; // TODO: unrequired result?-  remove
    fn ops_count(&self) -> usize;
}

pub trait ExecNow<D = Self> {
    fn exec_now(&self, device: &D, range_bounds: impl RangeBounds<usize>) -> crate::Result<()>;

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
        impl<Mods: $crate::AddOperation> $crate::AddOperation for $device<Mods> {
            #[inline]
            fn add_op<Args: $crate::Parents<N> + $crate::UpdateArgs<$crate::Buffers>, const N: usize>(
                &self,
                args: Args,
                operation: fn(&mut Args) -> crate::Result<()>,
            ) -> $crate::Result<()> {
                self.modules.add_op(args, operation)
            }

            #[inline]
            fn ops_count(&self) -> usize {
                self.modules.ops_count()
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
    #[track_caller]
    fn construct_unified_buf_from_cpu_buf<'a, T: 'static, S: Shape>(
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
                fn construct_unified_buf_from_cpu_buf<'a, T: 'static, S: Shape>(
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
        }
    };
}

#[cfg(feature = "autograd")]
pass_down_use_gpu_or_cpu!(Autograd);

#[cfg(feature = "lazy")]
pass_down_use_gpu_or_cpu!(Lazy);

pub trait UseGpuOrCpu {
    #[track_caller]
    fn use_cpu_or_gpu(
        &self,
        location: crate::HashLocation<'static>,
        input_lengths: &[usize],
        cpu_op: impl FnMut(),
        gpu_op: impl FnMut(),
    ) -> GpuOrCpuInfo;
}

#[cfg(feature = "graph")]
pub trait OptimizeMemGraph {
    fn optimize_mem_graph(
        &self,
        graph_translator: Option<&crate::modules::GraphTranslator>,
    ) -> crate::Result<()>;
}

#[macro_export]
macro_rules! pass_down_optimize_mem_graph {
    ($to_impl:ident) => {
        impl<Mods: $crate::OptimizeMemGraph> $crate::OptimizeMemGraph for $to_impl<Mods> {
            fn optimize_mem_graph(
                &self,
                graph_translator: Option<&crate::modules::GraphTranslator>,
            ) -> crate::Result<()> {
                self.modules.optimize_mem_graph(graph_translator)
            }
        }
    };
}
