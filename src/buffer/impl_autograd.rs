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
    pub fn grad_mut(&self) -> &'a mut Self
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
    pub fn try_grad_mut(&self) -> Option<&'a mut Self>
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
    pub fn grad_mut(&self) -> &'a mut Self {
        unimplemented!("Gradient not available. Activate the autograd feature.");
    }
}
