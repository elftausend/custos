use core::{fmt::Debug, hash::BuildHasherDefault, panic::Location};
use std::collections::HashMap;

use crate::{
    prelude::One, Alloc, Buffer, Device, Gradients, HasId, HashLocation, LocationHasher, Shape,
    WriteBuf,
};

// does not require the device param ???
pub type GradFn<'a> = Box<dyn Fn(&mut Gradients) + 'a>;

/// Stores the grad functions and gradient cache.
#[derive(Default)]
pub struct Tape<'a> {
    /// Caches gradients for each [`Buffer`]'s id ([`Ident`]).
    pub grads: Gradients,
    grad_fns: Vec<GradFn<'a>>,
    grad_fns_loc: HashMap<HashLocation<'static>, GradFn<'a>, BuildHasherDefault<LocationHasher>>,
    grad_fn_order: Vec<HashLocation<'static>>,
}

impl Debug for Tape<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Tape")
            .field("grads", &self.grads)
            .field("grad_fns", &self.grad_fns.len())
            .finish()
    }
}

impl<'a> Tape<'a> {
    /// Adds a gradient function to the tape.
    #[inline]
    #[track_caller]
    pub fn add_grad_fn<F: Fn(&mut Gradients) + 'a>(&mut self, grad_fn: F) {
        let hash_location = Location::caller().into();

        if self.grad_fns_loc.contains_key(&hash_location) {
            return;
        }

        self.grad_fns_loc.insert(hash_location, Box::new(grad_fn));
        self.grad_fn_order.push(hash_location)

        // self.grad_fns.push(Box::new(grad_fn))
    }

    /// Calls all gradient functions in reverse order.
    pub fn backward<D>(&mut self, _device: &D) {
        for grad_fn_id in self.grad_fn_order.iter().rev() {
            let grad_fn = self.grad_fns_loc.get(grad_fn_id).unwrap();
            grad_fn(&mut self.grads);
        }
        /*for grad_fn in self.grad_fns.drain(..).rev() {
            grad_fn(&mut self.grads);
        }*/
    }

    /// Backward pass with seeded gradient.
    /// The seed of the gradient contains `buf.len()` elements, all of them are set to 1.
    pub fn backward_seeded<T, D: Device, S: Shape>(&mut self, buf: &Buffer<T, D, S>)
    where
        T: Clone + One + 'static,
        D: Alloc<T> + WriteBuf<T, S, D> + 'static,
    {
        let out = self.grads.get_mut::<T, S, D>(buf.device(), buf.id());
        out.write(&vec![T::one(); out.len()]);

        self.backward(buf.device())
    }
}
