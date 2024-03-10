use crate::{
    AddOperation, Alloc, BoxedShallowCopy, Buffer, Buffers, HasId, LazyGraph, Parents, Shape,
    TapeActions, UpdateArgs, WriteBuf, ZeroGrad,
};
use core::fmt::Debug;

use super::Gradients;

pub type GradFn = Box<dyn Fn(&mut Gradients)>;

/// Stores the grad functions and gradient cache.
#[derive(Default)]
pub struct Tape {
    grad_fns: Vec<GradFn>,
    pub lazy_graph: LazyGraph<Box<dyn BoxedShallowCopy>>,
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
    pub fn add_grad_fn<Args: Parents<N> + UpdateArgs, const N: usize>(
        &mut self,
        args: Args,
        op: fn(&mut Args) -> crate::Result<()>,
    ) {
        self.lazy_graph.add_operation(args, op);
    }

    /// Calls all gradient functions in reverse order.
    pub fn backward(
        &mut self,
        buffers: &mut Buffers<Box<dyn BoxedShallowCopy>>,
        lazy_enabled: bool,
    ) {
        for val in self.lazy_graph.iter_with(buffers).rev() {
            val.unwrap();
        }
        if !lazy_enabled {
            self.lazy_graph.clear();
        }
    }

    pub fn seed_grad_for_buf<T, D, S>(&self, buf: &Buffer<T, D, S>, seed: &[T])
    where
        T: 'static,
        D: WriteBuf<T, S, D> + TapeActions + ZeroGrad<T> + Alloc<T> + 'static,
        S: Shape,
    {
        let gradients = unsafe { buf.device().gradients_mut() }.unwrap();

        let out = gradients.get_mut::<T, S, D>(buf.device(), buf.id());
        out.write(seed);
    }

    pub fn backward_seeded_with_buffers<T, D, S: Shape>(
        &mut self,
        buf: &Buffer<T, D, S>,
        seed: &[T],
        buffers: &mut Buffers<Box<dyn BoxedShallowCopy>>,
    ) where
        T: 'static,
        D: Alloc<T> + ZeroGrad<T> + WriteBuf<T, S, D> + TapeActions + AddOperation + 'static,
    {
        self.seed_grad_for_buf(buf, seed);

        let is_lazy_enabled = buf.device().is_lazy_enabled();
        buf.device()
            .eagerly(|| self.backward(buffers, is_lazy_enabled));
    }

    pub fn backward_seeded_maybe_with_buffers<T, D, S: Shape>(
        &mut self,
        buf: &Buffer<T, D, S>,
        seed: &[T],
        buffers: Option<&mut Buffers<Box<dyn BoxedShallowCopy>>>,
    ) where
        T: 'static,
        D: Alloc<T> + ZeroGrad<T> + WriteBuf<T, S, D> + TapeActions + AddOperation + 'static,
    {
        match buffers {
            Some(buffers) => self.backward_seeded_with_buffers(buf, seed, buffers),
            None => {
                let mut no_grads = {
                    // unique mutable access to gradients
                    let gradients = unsafe { buf.device().gradients_mut() }.unwrap();

                    let no_grads = &mut gradients.no_grads_pool;
                    core::mem::take(no_grads)

                    // ... destroy unique mutable access
                };

                // unique mutable access required for "buf.grad()"s in grad functions
                self.backward_seeded_with_buffers(buf, seed, &mut no_grads);

                let gradients = unsafe { buf.device().gradients_mut() }.unwrap();
                let no_grads_src = &mut gradients.no_grads_pool;
                *no_grads_src = no_grads;
            }
        }
    }
}
