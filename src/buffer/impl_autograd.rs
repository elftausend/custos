use crate::MayGradActions;
use crate::prelude::*;

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
    pub fn backward<'b>(&self) -> crate::Result<()>
    where
        T: Clone + One + 'static,
        D: TapeActions<'b>
            + ZeroGrad<T>
            + WriteBuf<T, S, D>
            + Alloc<T>
            + AddOperation
            + 'static
            + GradActions,
        D: CachedBuffers,
    {
        self.backward_with(&vec![T::one(); self.len()])
    }

    /// Calls `.backward_seeded_maybe_with_buffers` on the [`Tape`] with the given buffer.
    #[inline]
    pub fn backward_with<'b>(&self, seed: &[T]) -> crate::Result<()>
    where
        T: Clone + 'static,
        D: CachedBuffers
            + TapeActions<'b>
            + GradActions
            + ZeroGrad<T>
            + WriteBuf<T, S, D>
            + Alloc<T>
            + AddOperation
            + 'static,
    {
        // should never be None
        if let Some(mut tape) = self.device().tape_mut() {
            let mut buffers = unsafe { self.device().buffers_mut() };
            tape.backward_seeded_maybe_with_buffers(self, seed, buffers.as_deref_mut())
        } else {
            Ok(())
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
    pub fn grad(&self) -> &Buffer<'static, T, D, S>
    where
        D: ZeroGrad<T> + MayGradActions + Alloc<T>,
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
    /// Returns none either if the autograd feature is disabled, no tape was found (add [`Autograd`] module) or no gradient is allocated.
    // TODO: Maybe return Result with two error variants?
    pub fn try_grad(&self) -> Option<&'a Buffer<'static, T, D, S>>
    where
        D: MayGradActions + Alloc<T>,
    {
        if !self.requires_grad() {
            return None;
        }

        #[cfg(feature = "autograd")]
        unsafe {
            let device = self.device();
            device.gradients()?.may_get_ref(device, self.id()).ok()
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
    
    #[cfg(feature = "autograd")]
    #[inline]
    pub fn grad_mut_self<'b: 'c, 'c>(&'b mut self) -> (&'c Self, &'b mut Buffer<'static, T, D, S>)
    where
        D: GradActions + Alloc<T> + ZeroGrad<T>,
    {
        (self, unsafe { self.grad_mut_unbound() })
    }

    #[cfg(not(feature = "autograd"))]
    #[inline]
    pub fn grad_mut_self<'b: 'c, 'c>(&'b mut self) -> (&'c Self, &'b mut Buffer<'static, T, D, S>)
    where
        D: GradActions + Alloc<T> + ZeroGrad<T>,
    {
        unimplemented!("Gradient not available. Activate the autograd feature.");
    }

    #[cfg(feature = "autograd")]
    #[inline]
    pub fn grad_mut<'b>(&'b mut self) -> &'b mut Buffer<'static, T, D, S>
    where
        D: GradActions + Alloc<T> + ZeroGrad<T>,
    {
        unsafe { self.grad_mut_unbound() }
    }

    /// Returns a mutable reference to the gradient of this buffer.
    /// This allocates a gradient buffer if it wasn't previously.
    #[inline]
    #[cfg(feature = "autograd")]
    pub unsafe fn grad_mut_unbound<'b>(&'b self) -> &'a mut Buffer<'static, T, D, S>
    where
        D: GradActions + Alloc<T> + ZeroGrad<T>,
    {
        // TODO: consider activating this check ->
        // e.g. binary grad ops are computed in a single function where differentiating between
        // req grad and no req grad is not possible/ difficult
        // assert!(self.requires_grad(), "Buffer does not require gradient.");
        unsafe { self.device().grad_mut(self.device(), self) }
        // unsafe {
        //     self.device()
        //         .gradients_mut()
        //         .expect(AUTOGRAD_NOT_AVAILABLE)
        //         // .grads
        //         .get_mut(self.device(), self.id())
        // }
    }

    /// Returns a mutable reference to the gradient of this buffer.
    /// Returns none either if the autograd feature is disabled, no tape was found (add [`Autograd`] module) or no gradient is allocated.
    // TODO: Maybe return Result with two error variants?
    pub fn try_grad_mut(&mut self) -> Option<&mut Buffer<'static, T, D, S>>
    where
        D: MayGradActions + Alloc<T>,
    {
        if !self.requires_grad() {
            return None;
        }

        #[cfg(feature = "autograd")]
        unsafe {
            let device = self.device();
            device.gradients_mut()?.may_get_mut(device, self.id()).ok()
        }

        #[cfg(not(feature = "autograd"))]
        None
    }

    /// In this case, this is just a dummy function.
    /// Activate the `autograd` feature to make this function useable.
    #[inline]
    #[cfg(not(feature = "autograd"))]
    pub fn grad_mut<'b>(&'b self) -> &'b mut Self {
        unimplemented!("Gradient not available. Activate the autograd feature.");
    }

    /// In this case, this is just a dummy function.
    /// Activate the `autograd` feature to make this function useable.
    #[inline]
    #[cfg(not(feature = "autograd"))]
    pub unsafe fn grad_mut_unbound<'b>(&'b self) -> &'a mut Self {
        unimplemented!("Gradient not available. Activate the autograd feature.");
    }
}

mod tests {
    // use crate::Number;

    // fn run<T: Number>(x: &mut [T], y: &[T]) {
    //     for i in 0..x.len() {
    //         x[i] = y[i] + T::one()
    //     }
    // }

    // #[cfg(feature = "cpu")]
    // #[cfg(feature = "autograd")]
    // #[test]
    // fn test_multiple_grad_mut_comp_error() {
    //     use crate::{Autograd, Base, Cached, Device, CPU};

    //     let device = CPU::<Autograd<Cached<Base>>>::new();
    //     let mut buf = device.buffer([1, 2, 3, 4]);

    //     let _out = buf.grad();
    //     let out = buf.grad_mut();
    //     run(out, _out);
    // }
}
