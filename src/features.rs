use core::any::Any;

#[cfg(feature = "cached")] 
use core::cell::{Ref, RefMut};

use crate::{Parents, Shape, CPU};

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
    fn setup(_device: &mut D) {}
}

pub trait Run<D> {
    #[inline]
    fn run(&self, _device: &D) {}
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
        Self: Device + 'static,
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

pub trait Operation {
    fn forward(&mut self);
}

pub trait AddOperation {
    fn add_operation2(&self, _operation: impl Operation) {}
    unsafe fn add_operation<T: 'static, D: Device + 'static, S: Shape>(
        &self,
        out: &mut Buffer<T, D, S>,
        operation: impl Fn(&mut dyn Any),
    );
    fn call_lazily(&self) {}
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
macro_rules! impl_unified_mem_chain {
    ($($to_impl:ident),*) => {
        $(
            impl<Mods: UnifiedMemChain<D>, D: Device> UnifiedMemChain<D> for $to_impl<Mods> {
                fn construct_unified_buf_from_cpu_buf<'a, T, S: Shape>(
                    &self,
                    device: &'a D,
                    no_drop_buf: Buffer<'a, T, CachedCPU, S>
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
impl_unified_mem_chain!(Lazy);

#[cfg(feature = "autograd")]
impl_unified_mem_chain!(Autograd);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct GpuOrCpuInfo {
    pub use_cpu: bool,
    pub is_result_cached: bool,
}

macro_rules! impl_use_gpu_or_cpu {
    ($to_impl:ident) => {
        impl<Mods: UseGpuOrCpu> UseGpuOrCpu for $to_impl<Mods> {
            #[inline]
            fn use_cpu_or_gpu(
                &self,
                location: crate::HashLocation<'static>,
                input_lengths: &[usize],
                cpu_op: impl FnMut(),
                gpu_op: impl FnMut(),
            ) -> GpuOrCpuInfo {
                self.modules
                    .use_cpu_or_gpu(location, input_lengths, cpu_op, gpu_op)
            }
        }
    };
}

#[cfg(feature = "autograd")]
impl_use_gpu_or_cpu!(Autograd);

#[cfg(feature = "lazy")]
impl_use_gpu_or_cpu!(Lazy);

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
