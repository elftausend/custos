use core::{any::Any, fmt::Debug, hash::BuildHasherDefault, panic::Location};
use std::collections::{HashMap, HashSet};

use crate::{
    prelude::One, Alloc, Buffer, Buffers, HasId, HashLocation, LazyGraph, LocationHasher, Parents,
    Shape, TapeActions, UpdateArgs, WriteBuf,
};

use super::Gradients;

pub type GradFn = Box<dyn Fn(&mut Gradients)>;

/// Stores the grad functions and gradient cache.
#[derive(Default)]
pub struct Tape {
    // Caches gradients for each [`Buffer`]'s id ([`Ident`]).
    // pub grads: Gradients,
    grad_fns: Vec<GradFn>,
    grad_fns_loc: HashMap<HashLocation<'static>, GradFn, BuildHasherDefault<LocationHasher>>,
    grad_fn_order: Vec<HashLocation<'static>>,

    unconsumed_locations: HashSet<HashLocation<'static>>,

    pub lazy_graph: LazyGraph<Box<dyn Any>>,
}

impl Debug for Tape {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Tape")
            // .field("grads", &self.grads)
            .field("grad_fns", &self.grad_fns.len())
            .finish()
    }
}

impl Tape {
    #[inline]
    #[track_caller]
    pub fn add_grad_fn2<Args: Parents<N> + UpdateArgs, const N: usize>(
        &mut self,
        args: Args,
        op: fn(&mut Args) -> crate::Result<()>,
    ) {
        // if self
        //     .unconsumed_locations
        //     .contains(&Location::caller().into())
        // {
        //     return;
        // }
        self.lazy_graph.add_operation(args, op);
        // self.unconsumed_locations.insert(Location::caller().into());
    }

    /// Adds a gradient function to the tape.
    #[inline]
    #[track_caller]
    pub fn add_grad_fn<F: Fn(&mut Gradients) + 'static>(&mut self, grad_fn: F) {
        let hash_location = Location::caller().into();

        if self.grad_fns_loc.contains_key(&hash_location) {
            return;
        }

        self.grad_fns_loc.insert(hash_location, Box::new(grad_fn));
        self.grad_fn_order.push(hash_location)

        // self.grad_fns.push(Box::new(grad_fn))
    }

    /// Calls all gradient functions in reverse order.
    pub fn backward(&mut self, buffers: &mut Buffers<Box<dyn Any>>) {
        // for grad_fn_id in self.grad_fn_order.iter().rev() {
        //     let grad_fn = self.grad_fns_loc.get(grad_fn_id).unwrap();
        //     grad_fn(&mut self.grads);
        // }

        for val in self
            .lazy_graph
            // .iter_with(&mut self.grads.no_grads_pool.cache)
            .iter_with(buffers)
            .rev()
        {
            val.unwrap();
        }
        self.unconsumed_locations.clear();
        self.lazy_graph.clear(); /*for grad_fn in self.grad_fns.drain(..).rev() {
                                     grad_fn(&mut self.grads);
                                 }*/
    }

    /// Backward pass with seeded gradient.
    /// The seed of the gradient contains `buf.len()` elements, all of them are set to 1.
    pub fn backward_seeded<T, D, S: Shape>(&mut self, buf: &Buffer<T, D, S>)
    where
        T: Clone + One + 'static,
        D: Alloc<T> + WriteBuf<T, S, D> + TapeActions + 'static,
    {
        let mut no_grads = {
            // unique mutable access to gradients
            let gradients = unsafe { buf.device().gradients_mut() }.unwrap();

            let out = gradients.get_mut::<T, S, D>(buf.device(), buf.id());
            out.write(&vec![T::one(); out.len()]);

            let no_grads = &mut gradients.no_grads_pool.cache;
            core::mem::take(no_grads)

            // ... destroy unique mutable access
        };

        // unique mutable access required for "buf.grad()"s in grad functions
        self.backward(&mut no_grads);

        let gradients = unsafe { buf.device().gradients_mut() }.unwrap();
        let no_grads_src = &mut gradients.no_grads_pool.cache;
        *no_grads_src = no_grads;
    }
}
