#[cfg(feature = "cached")]
use core::cell::{Ref, RefMut};
use core::{fmt::Debug, ops::RangeBounds};

use crate::{HasId, Parents, Shape, TranslatedCacheTrace, UniqueId, CPU};

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
pub trait Retrieve<D, T>: OnDropBuffer {
    // "generator"
    #[track_caller]
    fn retrieve<S, const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        parents: impl Parents<NUM_PARENTS>,
    ) -> D::Data<T, S>
    where
        S: Shape,
        D: Device + Alloc<T>;

    // "actor"
    #[inline]
    fn on_retrieve_finish<S: Shape>(&self, _retrieved_buf: &Buffer<T, D, S>)
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

#[cfg(feature = "autograd")]
pub trait TapeActions {
    // "generator" - do not forget to pass down
    #[inline]
    fn tape(&self) -> Option<Ref<crate::Tape>> {
        None
    }
    // "generator" - do not forget to pass down
    #[inline]
    fn tape_mut(&self) -> Option<RefMut<crate::Tape>> {
        None
    }

    // use track caller to identify a specific grad function
    //-> if backward is not called (.drain()), the grad fn vector will gradually fill up
    #[track_caller]
    fn add_grad_fn(
        &self,
        // ids: impl AllocGradsFrom<N>,
        grad_fn: impl Fn(&mut crate::Gradients) + 'static,
    ) where
        // T: 'static,
        Self: 'static,
    {
        if let Some(mut tape) = self.tape_mut() {
            // the type T must match for every Id!
            // for id in ids.ids() {
            //     tape.grads.grads_pool.add_buf_once::<T, Self, S>(self, id)
            // }

            tape.add_grad_fn(grad_fn)
        }
    }
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

    // fn from_cache(cache: &std::collections::HashMap<UniqueId, ()>, ids: [UniqueId; 2]) -> Self {
    //     todo!()
    // }
}

pub trait AddOperation<T, D: Device> {
    fn add_op<S: Shape, Args: Parents<N>, const N: usize>(
        &self,
        args: Args,
        out: &mut Buffer<T, D, S>,
        operation: fn(&mut Buffer<T, D, S>, &Args) -> crate::Result<()>,
    ) -> crate::Result<()>;
    fn ops_count(&self) -> usize;
}

pub trait ExecNow<D = Self> {
    fn exec_now(&self, range_bounds: impl RangeBounds<usize>) -> crate::Result<()>;

    #[inline]
    fn exec_last_n(&self, last_n: usize) -> crate::Result<()>
    where
        D: Device,
        Self: AddOperation<i32, D>,
    {
        self.exec_now(self.ops_count() - last_n..)
    }
}

/// Implements the [`AddOperation`] trait for any supplied device. The `add_op` call is passed down to `self.modules`.
#[macro_export]
macro_rules! pass_down_add_operation {
    ($device:ident) => {
        impl<T, D: $crate::Device, Mods: $crate::AddOperation<T, D>> $crate::AddOperation<T, D>
            for $device<Mods>
        {
            #[inline]
            fn add_op<S: Shape, Args: $crate::Parents<N>, const N: usize>(
                &self,
                args: Args,
                out: &mut $crate::Buffer<T, D, S>,
                operation: fn(&mut Buffer<T, D, S>, &Args) -> crate::Result<()>,
            ) -> $crate::Result<()> {
                self.modules.add_op(args, out, operation)
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
                range_bounds: impl core::ops::RangeBounds<usize>,
            ) -> $crate::Result<()> {
                self.modules.exec_now(range_bounds)
            }
        }
    };
}

#[macro_export]
macro_rules! pass_down_exec_now {
    ($device:ident) => {
        impl<Mods: $crate::ExecNow<Self>> $crate::ExecNow<Self> for $device<Mods> {
            #[inline]
            fn exec_now(
                &self,
                range_bounds: impl core::ops::RangeBounds<usize>,
            ) -> $crate::Result<()> {
                self.modules.exec_now(range_bounds)
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
    fn construct_unified_buf_from_cpu_buf<'a, T, S: Shape>(
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
                fn construct_unified_buf_from_cpu_buf<'a, T, S: Shape>(
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

pub trait OptimizeMemGraph {
    fn optimize_mem_graph(
        &self,
        cache_traces: Option<&[TranslatedCacheTrace]>,
    ) -> crate::Result<()>;
}

#[macro_export]
macro_rules! pass_down_optimize_mem_graph {
    ($to_impl:ident) => {
        impl<Mods: $crate::OptimizeMemGraph> $crate::OptimizeMemGraph for $to_impl<Mods> {
            fn optimize_mem_graph(
                &self,
                cache_traces: Option<&[$crate::TranslatedCacheTrace]>,
            ) -> crate::Result<()> {
                self.modules.optimize_mem_graph(cache_traces)
            }
        }
    };
}
