use core::{
    any::Any,
    cell::{Ref, RefMut},
};

use crate::{Base, Cached, CachedModule, DeviceError, Parents, Shape, CPU};

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

pub trait HasModules<Mods> {
    fn modules(&self) -> &Mods;
}

#[cfg(feature = "autograd")]
pub trait TapeActions<D> {
    // "generator" - do not forget to pass down
    #[inline]
    fn tape(&self) -> Option<Ref<crate::Tape<D>>> {
        None
    }
    // "generator" - do not forget to pass down
    #[inline]
    fn tape_mut(&self) -> Option<RefMut<crate::Tape<D>>> {
        None
    }

    // use track caller to identify a specific grad function
    //-> if backward is not called (.drain()), the grad fn vector will gradually fill up
    #[track_caller]
    fn add_grad_fn</*T, S: Shape*/>(
        &self,
        // ids: impl AllocGradsFrom<N>,
        grad_fn: impl Fn(&mut crate::Gradients, &D) + 'static,
    ) where
        D: Device,
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

pub type CachedCPU = CPU<CachedModule<Base, CPU<Cached<Base>>>>;

pub trait UnifiedMemChain<D: Device> {
    #[track_caller]
    fn construct_unified_buf_from_cpu_buf<'a, T, S: Shape>(
        &self,
        device: &'a D,
        no_drop_buf: Buffer<'a, T, CachedCPU, S>,
    ) -> crate::Result<Buffer<'a, T, D, S>>;
}

#[macro_export]
macro_rules! impl_unified_mem_chain {
    ($($to_impl:ident),*) => {
        $(
            impl<Mods: UnifiedMemChain<D>, D: Device> UnifiedMemChain<D> for $to_impl<Mods> {
                fn construct_unified_buf_from_cpu_buf<'a, T, S: Shape>(
                    &self,
                    device: &'a D,
                    no_drop_buf: Buffer<'a, T, CachedCPU, S>
                ) -> crate::Result<Buffer<'a, T, D, S>>
                {
                    self.modules.construct_unified_buf_from_cpu_buf(device, no_drop_buf)
                }
            }

        )*
    };
}

#[cfg(feature = "lazy")]
use crate::Lazy;

#[cfg(feature = "lazy")]
impl_unified_mem_chain!(Lazy);
