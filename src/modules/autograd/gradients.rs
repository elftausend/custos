use core::{any::Any, hash::BuildHasherDefault};
use std::collections::HashMap;

use crate::{
    Alloc, BorrowCache, BoxedShallowCopy, Buffer, Buffers, CachingError, Device, HasId, Id,
    NoHasher, Shape, UniqueId, ZeroGrad,
};

const INVALID_ID: &str = "A matching Buffer does not exist.";

/// A cache for gradients.
/// The cache is populated by `get_ref`, `get_like` or `get_mut_ref` calls.
#[derive(Default)]
pub struct Gradients {
    pub grads_pool: BorrowCache,
    pub no_grads_pool: Buffers<Box<dyn BoxedShallowCopy>>,
    pub zero_grad_cbs: Vec<(Id, fn(&mut dyn Any))>,
    pub buf_requires_grad: HashMap<UniqueId, bool, BuildHasherDefault<NoHasher>>,
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

            // the callback is only added if the grad buffer was used in a grad op, so this check should not be necessary (but it is)
            let req_grad = *self.buf_requires_grad.get(id).unwrap_or(&true);
            if req_grad {
                cb(&mut **grad_buf);
            }
        }
        // self.grads_pool.cache.clear();
    }

    pub fn add_zero_grad_cb<T, D, S>(&mut self, id: &Id)
    where
        T: 'static,
        D: Device + ZeroGrad<T> + 'static,
        S: Shape,
    {
        self.zero_grad_cbs.push((*id, |grad_buf| {
            let grad_buf = grad_buf.downcast_mut::<Buffer<T, D, S>>().unwrap();

            grad_buf.device().zero_grad(grad_buf);
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
            .ok_or(CachingError::InvalidId)
            .expect(INVALID_ID)
            .as_any()
            .downcast_ref()
            .ok_or(CachingError::InvalidTypeInfo)
            .expect(INVALID_ID)
    }
}

#[cfg(test)]
mod tests {

    #[cfg(feature = "cpu")]
    #[test]
    fn test_zero_grad_on_gradients() {
        use crate::{Base, Device, Gradients, HasId, CPU};

        let dev = CPU::<Base>::new();

        let mut grads = Gradients::default();

        let lhs = dev.buffer([1, 2, 3, 4]);
        {
            let grad = grads.get_mut::<i32, (), _>(&dev, lhs.id());

            for val in grad.iter_mut() {
                *val = 4;
            }

            assert_eq!(grad.as_slice(), &[4; 4]);
        }

        grads.zero_grad();
        let grad = grads.get_ref::<i32, (), _>(&dev, lhs.id());
        assert_eq!(grad.as_slice(), &[0; 4]);
    }
}
