use core::any::Any;

use crate::{Alloc, AnyBuffers, BorrowCache, Buffer, CachingError, Device, HasId, Id, Shape, ZeroGrad};

const INVALID_ID: &str = "A matching Buffer does not exist.";

/// A cache for gradients.
/// The cache is populated by `get_ref`, `get_like` or `get_mut_ref` calls.
#[derive(Default)]
pub struct Gradients {
    pub grads_pool: BorrowCache,
    pub no_grads_pool: AnyBuffers,
    pub zero_grad_cbs: Vec<(Id, fn(&mut dyn Any, &dyn Any))>,
}

impl core::fmt::Debug for Gradients {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Gradients")
            // .field("cache", &self.grads_pool)
            .finish()
    }
}

impl Gradients {
    /// Clears the cache.
    #[inline]
    pub fn zero_grad(&mut self) {
        for (id, cb) in &self.zero_grad_cbs {
            let grad_buf = self.grads_pool.cache.get_mut(id).unwrap();
            let buf = self.no_grads_pool.get_mut(id).unwrap();
            cb(&mut **grad_buf, &**buf);
        }
        // self.grads_pool.cache.clear();
    }

    pub fn add_zero_grad_cb<T: 'static, D: Device + ZeroGrad<T> + 'static, S: Shape>(
        &mut self,
        id: &Id,
    ) {
        self.zero_grad_cbs.push((*id, |grad_buf, buf| {
            let grad_buf = grad_buf.downcast_mut::<Buffer<T, D, S>>().unwrap();
            let buf = buf.downcast_ref::<Buffer<T, D, S>>().unwrap();

            // the callback is only added if the grad buffer was used in a grad op, so this check should not be necessary (but it is)
            if buf.requires_grad() {
                grad_buf.device().zero_grad(grad_buf);
            }
        }));
    }

    /// May get a reference to a gradient [`Buffer`].
    #[inline]
    pub fn may_get_ref<'a, T, S, D>(&self, ident: Id) -> Result<&Buffer<'a, T, D, S>, CachingError>
    where
        T: 'static,
        S: Shape,
        D: Alloc<T> + 'static,
    {
        self.grads_pool.get_buf(ident)
    }

    /// May get a mutable reference to a gradient [`Buffer`].
    #[inline]
    pub fn may_get_mut<'a, T, S, D>(
        &mut self,
        id: Id,
    ) -> Result<&mut Buffer<'a, T, D, S>, CachingError>
    where
        T: 'static,
        S: Shape,
        D: Alloc<T> + 'static,
    {
        self.grads_pool.get_buf_mut(id)
    }

    /// Returns a reference to a gradient [`Buffer`].
    /// Allocates a gradient [`Buffer`] if it does not exist.
    #[inline]
    pub fn get_ref<'a, T, S, D>(&mut self, device: &'a D, id: Id) -> &Buffer<'a, T, D, S>
    where
        T: 'static,
        S: Shape,
        D: Alloc<T> + ZeroGrad<T> + 'static,
    {
        // because of rust, thx
        let mut new_buf = false;
        self.grads_pool
            .add_buf_once::<T, D, S>(device, id, &mut new_buf);

        if new_buf {
            self.add_zero_grad_cb::<T, D, S>(&id);
        }
        self.grads_pool.get_buf(id).unwrap()
    }

    /// Returns a mutable reference to a gradient [`Buffer`].
    /// Allocates a gradient [`Buffer`] if it does not exist.
    #[inline]
    pub fn get_mut<'a, T, S, D>(&mut self, device: &'a D, id: Id) -> &mut Buffer<'a, T, D, S>
    where
        T: 'static,
        S: Shape,
        D: ZeroGrad<T> + Alloc<T> + 'static,
    {
        let mut new_buf = false;
        self.grads_pool
            .add_buf_once::<T, D, S>(device, id, &mut new_buf);

        if new_buf {
            self.add_zero_grad_cb::<T, D, S>(&id);
        }
        self.grads_pool.get_buf_mut(id).unwrap()
    }

    /// Returns a reference to a gradient [`Buffer`] using information from `buf`.
    #[inline]
    pub fn get_like<'a, T, S, D>(&mut self, buf: &Buffer<'a, T, D, S>) -> &Buffer<'a, T, D, S>
    where
        T: 'static,
        S: Shape,
        D: Alloc<T> + ZeroGrad<T> + 'static,
        D::Data<T, S>: HasId,
    {
        self.get_ref(buf.device(), buf.id())
    }

    #[inline]
    pub fn get_buf_from_no_grad_pool<'a, T, S, D>(&self, id: Id) -> &Buffer<'a, T, D, S>
    where
        T: 'static,
        S: Shape,
        D: Alloc<T> + 'static,
    {
        self.no_grads_pool
            .get(&id)
            .ok_or(CachingError::InvalidId).expect(INVALID_ID)
            .downcast_ref()
            .ok_or(CachingError::InvalidTypeInfo).expect(INVALID_ID)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Autograd, Base, Buffer, HasId, Retriever, CPU};

    #[test]
    #[cfg(feauture = "cpu")]
    fn test_same_types_get_double_return() {
        let device = CPU::<Autograd<Base>>::new();

        // let mut gradients = Gradients::default();

        let buf = Buffer::<i32, _>::new(&device, 10);
        // unsafe { register_buf(&mut gradients.no_grads_pool.borrow_mut().cache, &buf) }

        let out: Buffer<i32, _> = device.retrieve::<0>(buf.len(), ());
        // unsafe { register_buf(&mut gradients.no_grads_pool.borrow_mut().cache, &out) }

        // unsafe {
        //     (*device.modules.tape.get())
        //         .grads
        //         .get_double::<i32, (), (), CPU<Autograd<crate::CachedModule<Base, CPU>>>>((
        //             buf.id(),
        //             out.id(),
        //         ));
        // }

        // device
        //     .modules
        //     .tape
        //     .borrow_mut()
        //     .grads
        //     .get_double::<i32, (), (), CPU<Autograd<crate::CachedModule<Base, CPU>>>>((
        //         buf.id(),
        //         out.id(),
        //     ));
    }

    #[test]
    #[cfg(feauture = "cpu")]
    #[ignore = "deprecated"]
    #[should_panic]
    fn test_different_types_get_double_return() {
        let device = CPU::<Autograd<Base>>::new();

        // let mut gradients = Gradients::default();

        let buf = Buffer::<i32, _>::new(&device, 10);
        // unsafe { register_buf(&mut gradients.no_grads_pool.borrow_mut().cache, &buf) }

        let out: Buffer<i64, _> = device.retrieve::<0>(buf.len(), ());
        // unsafe { register_buf(&mut gradients.no_grads_pool.borrow_mut().cache, &out) }

        // unsafe {
        //     (*device.modules.tape.get()).grads
        //     .get_double::<i32, (), (), CPU<Autograd<crate::CachedModule<Base, CPU<Autograd<Base>>>>>>((buf.id(), out.id()));
        // }

        // unsafe {
        //     device
        //     .modules
        //     .tape
        //     .borrow_mut()
        //     .grads
        //     .get_double::<i32, (), (), CPU<Autograd<crate::CachedModule<Base, CPU<Autograd<Base>>>>>>((buf.id(), out.id()))
        // }
    }
}
