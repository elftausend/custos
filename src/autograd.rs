//! Provides tools for automatic differentiation.

use core::{
    cell::{Ref, RefMut},
    fmt::Debug,
    marker::PhantomData,
};

use crate::{
    borrowing_cache::BorrowingCache, prelude::One, Alloc, Buffer, Device, Ident, Shape, WriteBuf,
};

/// A cache for gradients.
/// The cache is populated by `get_ref`, `get_like` or `get_mut_ref` calls.
#[derive(Default)]
pub struct Gradients<D> {
    // maybe use a borrowed cache in the style of the 'owned' cache
    cache: BorrowingCache,
    _pd: PhantomData<D>,
}

impl<D> Debug for Gradients<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Gradients")
            .field("cache", &self.cache)
            .finish()
    }
}

type LhsRhsOut<'a, 'b, T, D, S> = (
    Buffer<'a, T, D, S>,
    Buffer<'a, T, D, S>,
    &'b mut Buffer<'a, T, D, S>,
    &'b mut Buffer<'a, T, D, S>,
    &'b Buffer<'a, T, D, S>,
);

impl<D> Gradients<D> {
    // everything is T, bad
    /*pub fn grads<'a, T>(&mut self, device: &'a D) -> Vec<Buffer<'a, T, D>> {
        self.cache
            .nodes
            .iter()
            .map(|(id, raw)| Buffer {
                ptr: D::destruct::<T, ()>(raw),
                device: Some(device),
                ident: *id,
            })
            .collect::<Vec<Buffer<T, D>>>()
    }*/

    /// Clears the cache.
    #[inline]
    pub fn zero_grad(&mut self) {
        self.cache.cache.clear();
    }

    /// May get a reference to a gradient [`Buffer`].
    #[inline]
    pub fn may_get_ref<'a, T, S>(&self, ident: Ident) -> Option<&Buffer<'a, T, D, S>>
    where
        T: 'static,
        S: Shape,
        D: Device + 'static,
    {
        self.cache.get_buf(ident)
    }

    /// May get a mutable reference to a gradient [`Buffer`].
    #[inline]
    pub fn may_get_mut<'a, T, S>(&mut self, ident: Ident) -> Option<&mut Buffer<'a, T, D, S>>
    where
        T: 'static,
        S: Shape,
        D: Device + 'static,
    {
        self.cache.get_buf_mut(ident)
    }

    /// Returns a reference to a gradient [`Buffer`].
    /// Allocates a gradient [`Buffer`] if it does not exist.
    #[inline]
    pub fn get_ref<'a, T, S>(&mut self, device: &'a D, ident: Ident) -> &Buffer<'a, T, D, S>
    where
        T: 'static,
        S: Shape,
        D: Alloc<'a, T, S> + 'static,
    {
        self.cache.add_or_get(device, ident)
    }

    /// Returns a mutable reference to a gradient [`Buffer`].
    /// Allocates a gradient [`Buffer`] if it does not exist.
    #[inline]
    pub fn get_mut<'a, T, S>(&mut self, device: &'a D, ident: Ident) -> &mut Buffer<'a, T, D, S>
    where
        T: 'static,
        S: Shape,
        D: for<'b> Alloc<'b, T, S> + 'static,
    {
        self.cache.add_or_get_mut(device, ident)
    }

    /// Returns a reference to a gradient [`Buffer`] using information from `buf`.
    #[inline]
    pub fn get_like<'a, T, S>(&mut self, buf: &Buffer<'a, T, D, S>) -> &Buffer<'a, T, D, S>
    where
        T: 'static,
        S: Shape,
        D: Alloc<'a, T, S> + 'static,
    {
        self.get_ref(buf.device(), buf.id())
    }

    /// Returns the forward [`Buffer`]s lhs and and rhs, and the gradient `Buffer`s lhs_grad, rhs_grad and out_grad.
    /// Usefull for binary operations.
    #[inline]
    pub fn get_triple<'a, T, S>(
        &mut self,
        device: &'a D,
        (lid, rid, oid): (Ident, Ident, Ident),
    ) -> LhsRhsOut<'a, '_, T, D, S>
    where
        T: 'static,
        S: Shape,
        D: for<'b> Alloc<'b, T, S> + 'static,
    {
        self.cache.add_buf_once(device, rid);
        self.cache.add_buf_once(device, oid);

        let lhs_grad_ptr = self.get_mut(device, lid) as *mut _;
        let lhs_grad = unsafe { &mut *lhs_grad_ptr };

        let rhs_grad_ptr = self.get_mut(device, rid) as *mut _;
        let rhs_grad = unsafe { &mut *rhs_grad_ptr };
        (
            unsafe { device.get_existing_buf(lid) },
            unsafe { device.get_existing_buf(rid) },
            lhs_grad,
            rhs_grad,
            self.may_get_ref(oid).unwrap(),
        )
    }

    /// Returns the forward [`Buffer`] x and the gradient `Buffer`s x_grad and out_grad.
    /// Usefull for unary operations.
    #[inline]
    pub fn get_double<'a, T, IS, OS>(
        &mut self,
        device: &'a D,
        (xid, oid): (Ident, Ident),
    ) -> (
        Buffer<'a, T, D, IS>,
        &mut Buffer<'a, T, D, IS>,
        &Buffer<'a, T, D, OS>,
    )
    where
        T: 'static,
        IS: Shape,
        OS: Shape,
        D: for<'b> Alloc<'b, T, IS> + for<'b> Alloc<'b, T, OS> + 'static,
    {
        let x_grad_ptr = self.get_mut(device, xid) as *mut _;
        let x_grad_mut = unsafe { &mut *x_grad_ptr };
        let o_grad = self.get_ref(device, oid);

        (unsafe { device.get_existing_buf(xid) }, x_grad_mut, o_grad)
    }
}

type GradFn<D> = Box<dyn Fn(&mut Gradients<D>, &D)>;

/// Stores the grad functions and gradient cache.
#[derive(Default)]
pub struct Tape<D: Device> {
    pub grads: Gradients<D>,
    grad_fns: Vec<GradFn<D>>,
}

/// This trait is implemented for all devices that provide a [`Tape`].
pub trait TapeReturn: Device {
    fn tape(&self) -> Ref<Tape<Self>>;
    fn tape_mut(&self) -> RefMut<Tape<Self>>;
}

impl<D: Device> Debug for Tape<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Tape").field("grads", &self.grads).finish()
    }
}

impl<D: Device> Tape<D> {
    /// Adds a gradient function to the tape.
    #[inline]
    pub fn add_grad_fn<F: Fn(&mut Gradients<D>, &D) + 'static>(&mut self, grad_fn: F) {
        self.grad_fns.push(Box::new(grad_fn))
    }

    /// Calls all gradient functions in reverse order.
    pub fn backward(&mut self, device: &D) {
        for grad_fn in self.grad_fns.drain(..).rev() {
            grad_fn(&mut self.grads, device);
        }
    }
    /// Backward pass with seeded gradient.
    /// The seed of the gradient contains `buf.len()` elements, all of them are set to 1.
    pub fn backward_seeded<T, S: Shape>(&mut self, buf: &Buffer<T, D, S>)
    where
        T: Clone + One + 'static,
        D: for<'a> Alloc<'a, T, S> + WriteBuf<T, S, D> + 'static,
    {
        // TODO // TODO
        //let mut out = self.grads.get_like::<T, S>(buf);
        let out = self.grads.get_mut::<T, S>(buf.device(), buf.id());
        out.write(&vec![T::one(); out.len()]);

        self.backward(buf.device())
    }
}

impl<'a, T, D, S> Buffer<'a, T, D, S>
where
    T: Clone + One + 'static,
    D: TapeReturn + WriteBuf<T, S, D> + for<'b> Alloc<'b, T, S> + 'static,
    S: Shape,
{
    /// Calls `.backward_seeded` on the [`Tape`].
    /// The seed of the gradient is set to `1` and contains `self.len()` elements.
    #[inline]
    pub fn backward(&self) {
        self.device().tape_mut().backward_seeded(self)
    }

    /// Returns a reference to the gradient of this buffer.
    /// The lifetime is bound to the lifetime of self, which is more strict.
    /// If the borrow checker complains, use `grad_unbound` instead.
    #[inline]
    pub fn grad(&self) -> Ref<Self> {
        Ref::map(self.device().tape(), |tape| {
            tape.grads.may_get_ref(self.id()).unwrap()
        })
    }

    /// Returns a reference to the gradient of this buffer.
    /// Lifetimes are checked during runtime.
    #[inline]
    pub fn grad_unbound(&self) -> Ref<'a, Self> {
        Ref::map(self.device().tape(), |tape| {
            tape.grads.may_get_ref(self.id()).unwrap()
        })
    }

    /// Returns a mutable reference to the gradient of this buffer.
    /// The lifetime is bound to the lifetime of self, which is more strict.
    /// If the borrow checker complains, use `grad_mut_unbound` instead.
    #[inline]
    pub fn grad_mut(&mut self) -> RefMut<Self> {
        RefMut::map(self.device().tape_mut(), |tape| {
            // TODO unwrap?, result?, try?, expect?
            tape.grads.may_get_mut(self.id()).unwrap()
        })
    }

    /// Returns a mutable reference to the gradient of this buffer.
    /// Lifetimes are checked during runtime.
    #[inline]
    pub fn grad_mut_unbound(&mut self) -> RefMut<'a, Self> {
        RefMut::map(self.device().tape_mut(), |tape| {
            // TODO unwrap?, result?, try?, expect?
            tape.grads.may_get_mut(self.id()).unwrap()
        })
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cpu")]
    #[cfg(feature = "macros")]
    use crate::{Buffer, Combiner};

    #[cfg(feature = "cpu")]
    #[cfg(feature = "macros")]
    #[test]
    fn test_tape_unary_ew() {
        use crate::{UnaryElementWiseMayGrad, CPU};

        let device = CPU::new();
        //let device = CPU::new();

        let buf = Buffer::from((&device, [1., -2., 3., -4., 5., 6.]));

        let out = device.unary_ew(&buf, |x| x.geq(0.).mul(x), |x| x.geq(0.));
        assert_eq!(out.read(), vec![1., 0., 3., 0., 5., 6.,]);

        out.backward();

        let grad = buf.grad();
        assert_eq!(grad.read(), vec![1., 0., 1., 0., 1., 1.,]);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_tape_unary_ew_cl() -> crate::Result<()> {
        use crate::{Buffer, OpenCL, UnaryElementWiseMayGrad, Combiner};

        let device = OpenCL::new(0)?;
        //let device = CPU::new();

        let buf = Buffer::from((&device, [1., -2., 3., -4., 5., 6.]));

        let out = device.unary_ew(&buf, |x| x.geq(0.).mul(x), |x| x.geq(0.));
        assert_eq!(out.read(), vec![1., 0., 3., 0., 5., 6.,]);

        out.backward();

        let grad = buf.grad();
        assert_eq!(grad.read(), vec![1., 0., 1., 0., 1., 1.,]);

        Ok(())
    }
}
