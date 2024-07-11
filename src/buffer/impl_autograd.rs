use crate::prelude::*;
use crate::MayTapeActions;

use crate::Unit;
#[cfg(feature = "autograd")]
use crate::ZeroGrad;

#[cfg(feature = "autograd")]
const AUTOGRAD_NOT_AVAILABLE: &str = "Autograd<> is not available.";

#[cfg(feature = "autograd")]
impl<'a, T, D, S> Buffer<'a, T, D, S>
where
    T: Unit + 'static,
    D: HasAutograd + Device + 'static,
    S: Shape,
{
    /// Calls `.backward_seeded` on the [`Tape`].
    #[inline]
    pub fn backward(&self)
    where
        T: Clone + One + 'static,
        D: TapeActions + ZeroGrad<T> + WriteBuf<T, S, D> + Alloc<T> + AddOperation + 'static,
        D: CachedBuffers,
    {
        self.backward_with(&vec![T::one(); self.len()]);
    }

    /// Calls `.backward_seeded` on the [`Tape`].
    #[inline]
    pub fn backward_lt(&self)
    where
        T: Clone + One + 'static,
        D: TapeActionsLT<'a> + ZeroGrad<T> + WriteBuf<T, S, D> + Alloc<T> + AddOperation + 'static,
        D: CachedBuffers,
    {
        self.backward_with_lt(&vec![T::one(); self.len()]);
    }

    /// Calls `.backward_seeded_maybe_with_buffers` on the [`Tape`] with the given buffer.
    #[inline]
    pub fn backward_with(&self, seed: &[T])
    where
        T: Clone + 'static,
        D: CachedBuffers
            + TapeActions
            + ZeroGrad<T>
            + WriteBuf<T, S, D>
            + Alloc<T>
            + AddOperation
            + 'static,
    {
        // should never be None
        if let Some(tape) = unsafe { self.device().tape_mut() } {
            let mut buffers = unsafe { self.device().buffers_mut() };
            tape.backward_seeded_maybe_with_buffers(self, seed, buffers.as_deref_mut())
        }
    }

    /// Calls `.backward_seeded_maybe_with_buffers` on the [`Tape`] with the given buffer.
    #[inline]
    pub fn backward_with_lt(&self, seed: &[T])
    where
        T: Clone + 'static,
        D: CachedBuffers
            + TapeActionsLT<'a>
            + ZeroGrad<T>
            + WriteBuf<T, S, D>
            + Alloc<T>
            + AddOperation
            + 'static,
    {
        // should never be None
        if let Some(tape) = unsafe { self.device().tape_mut() } {
            let mut buffers = unsafe { self.device().buffers_mut() };
            tape.backward_seeded_maybe_with_buffers_lt(self, seed, buffers.as_deref_mut())
        }
    }
}

impl<'a, T, D, S> Buffer<'a, T, D, S>
where
    T: Unit + 'static,
    D: Device + 'static,
    S: Shape,
{
    /// Returns a reference to the gradient of this buffer.
    /// This allocates a gradient buffer if it wasn't previously.
    ///
    /// Panics if the gradient was not allocated.
    #[inline]
    #[cfg(feature = "autograd")]
    pub fn grad_lt(&self) -> &'a Self
    where
        D: ZeroGrad<T> + crate::MayTapeActionsLT<'a> + Alloc<T>,
    {
        // TODO: consider activating this check ->
        // e.g. binary grad ops are computed in a single function where differentiating between
        // req grad and no req grad is not possible/ difficult
        // assert!(self.requires_grad(), "Buffer does not require gradient.");
        unsafe {
            self.device()
                .gradients_mut()
                .expect(AUTOGRAD_NOT_AVAILABLE)
                // .grads
                .get_ref(self.device(), self.id())
        }
    }
}

impl<'a, T, D, S> Buffer<'a, T, D, S>
where
    T: Unit + 'static,
    D: Device + 'static,
    S: Shape,
{
    /// Returns a reference to the gradient of this buffer.
    /// This allocates a gradient buffer if it wasn't previously.
    ///
    /// Panics if the gradient was not allocated.
    #[inline]
    #[cfg(feature = "autograd")]
    pub fn grad(&self) -> &'a Self
    where
        D: ZeroGrad<T> + MayTapeActions + Alloc<T>,
    {
        // TODO: consider activating this check ->
        // e.g. binary grad ops are computed in a single function where differentiating between
        // req grad and no req grad is not possible/ difficult
        // assert!(self.requires_grad(), "Buffer does not require gradient.");
        unsafe {
            self.device()
                .gradients_mut()
                .expect(AUTOGRAD_NOT_AVAILABLE)
                // .grads
                .get_ref(self.device(), self.id())
        }
    }

    /// Returns a reference to the gradient of this buffer.
    /// Returns none either if the autograd feature is disabled, no tape was found (add [`Autograd`] module) or no gradient was allocated previously.
    // TODO: Maybe return Result with two error variants?
    pub fn try_grad(&self) -> Option<&'a Self>
    where
        D: MayTapeActions + Alloc<T>,
    {
        if !self.requires_grad() {
            return None;
        }

        #[cfg(feature = "autograd")]
        unsafe {
            self.device().gradients()?.may_get_ref(self.id()).ok()
        }

        #[cfg(not(feature = "autograd"))]
        None
    }

    /// In this case, this is just a dummy function.
    /// Activate the `autograd` feature to make this function useable.
    #[inline]
    #[cfg(not(feature = "autograd"))]
    pub fn grad(&self) -> &'a Self {
        unimplemented!("Gradient not available. Activate the autograd feature.");
    }

    /// Returns a mutable reference to the gradient of this buffer.
    /// This allocates a gradient buffer if it wasn't previously.
    #[inline]
    #[cfg(feature = "autograd")]
    pub unsafe fn grad_mut<'b>(&'b self) -> &'a mut Self
    where
        D: MayTapeActions + Alloc<T> + ZeroGrad<T>,
    {
        // TODO: consider activating this check ->
        // e.g. binary grad ops are computed in a single function where differentiating between
        // req grad and no req grad is not possible/ difficult
        // assert!(self.requires_grad(), "Buffer does not require gradient.");
        unsafe {
            self.device()
                .gradients_mut()
                .expect(AUTOGRAD_NOT_AVAILABLE)
                // .grads
                .get_mut(self.device(), self.id())
        }
    }

    /// Returns a mutable reference to the gradient of this buffer.
    /// Returns none either if the autograd feature is disabled, no tape was found (add [`Autograd`] module) or no gradient was allocated previously.
    // TODO: Maybe return Result with two error variants?
    pub fn try_grad_mut<'b>(&'b mut self) -> Option<&'a mut Self>
    where
        D: MayTapeActions + Alloc<T>,
    {
        if !self.requires_grad() {
            return None;
        }

        #[cfg(feature = "autograd")]
        unsafe {
            self.device().gradients_mut()?.may_get_mut(self.id()).ok()
        }

        #[cfg(not(feature = "autograd"))]
        None
    }

    /// In this case, this is just a dummy function.
    /// Activate the `autograd` feature to make this function useable.
    #[inline]
    #[cfg(not(feature = "autograd"))]
    pub fn grad_mut<'b>(&'b mut self) -> &'a mut Self {
        unimplemented!("Gradient not available. Activate the autograd feature.");
    }
}

mod tests {
    use crate::Number;

    fn run<T: Number>(x: &mut [T], y: &mut [T]) {
        for i in 0..x.len() {
            x[i] = y[i] + T::one()
        }
    }

    #[cfg(feature = "cpu")]
    #[cfg(feature = "autograd")]
    #[test]
    fn test_multiple_grad_mut() {
        use crate::{Autograd, AutogradLT, Base, Cached, Device, CPU};

        let device = CPU::<Autograd<Cached<Base>>>::new();
        let mut buf = device.buffer([1, 2, 3, 4]);
        let _out = buf.grad();
        // let out = buf.grad_mut();
        // run(_out, out);
    }
}
