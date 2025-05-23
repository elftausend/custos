use core::{any::Any, hash::BuildHasherDefault};
use std::collections::HashMap;

use crate::{
    Alloc, BorrowCache, BoxedShallowCopy, Buffer, Buffers, CachingError, Device, HasId, Id,
    NoHasher, Shape, UniqueId, Unit, ZeroGrad,
};

/// A cache for gradients.
/// The cache is populated by `get_ref`, `get_like` or `get_mut_ref` calls.
#[derive(Default)]
pub struct Gradients {
    pub(crate) grads_pool: BorrowCache,
    pub no_grads_pool: Buffers<Box<dyn BoxedShallowCopy>>,
    zero_grad_cbs: Vec<(Id, fn(&mut dyn Any, &dyn Any))>,
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
    pub fn zero_grad<D: 'static>(&mut self, device: &D) {
        let device: &dyn Any = device;
        for (id, cb) in &self.zero_grad_cbs {
            let grad_buf = self.grads_pool.cache.get_mut(id).unwrap();

            // the callback is only added if the grad buffer was used in a grad op, so this check should not be necessary (but it is)
            let req_grad = *self.buf_requires_grad.get(id).unwrap_or(&true);
            if req_grad {
                cb(&mut **grad_buf, device);
            }
        }
        // self.grads_pool.cache.clear();
    }

    #[inline]
    pub fn add_zero_grad_cb<T, D, S>(&mut self, id: &Id)
    where
        T: Unit + 'static,
        D: Device + ZeroGrad<T> + 'static,
        S: Shape,
    {
        self.zero_grad_cbs.push((*id, |grad_buf, dev: &dyn Any| {
            let grad_buf = grad_buf.downcast_mut::<Buffer<T, D, S>>().unwrap();
            let device = dev.downcast_ref::<D>().unwrap();

            device.zero_grad(grad_buf);
        }));
    }

    /// May get a reference to a gradient [`Buffer`].
    #[inline]
    pub(crate) fn may_get_ref<'a, T, S, D>(
        &self,
        device: &'a D,
        ident: Id,
    ) -> Result<&Buffer<'static, T, D, S>, CachingError>
    where
        T: Unit + 'static,
        S: Shape,
        D: Alloc<T> + 'static,
    {
        self.grads_pool.get_buf(device, ident)
    }

    /// May get a mutable reference to a gradient [`Buffer`].
    #[inline]
    pub(crate) unsafe fn may_get_mut<'a, T, S, D>(
        &mut self,
        device: &'a D,
        id: Id,
    ) -> Result<&mut Buffer<'static, T, D, S>, CachingError>
    where
        T: Unit + 'static,
        S: Shape,
        D: Alloc<T> + 'static,
    {
        self.grads_pool.get_buf_mut(device, id)
    }

    /// Returns a reference to a gradient [`Buffer`].
    /// Allocates a gradient [`Buffer`] if it does not exist.
    #[inline]
    pub fn get_ref<'a, T, S, D>(&mut self, device: &'a D, id: Id) -> &Buffer<'static, T, D, S>
    where
        T: Unit + 'static,
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
        self.grads_pool.get_buf_mut(device, id).unwrap()
    }

    /// Returns a mutable reference to a gradient [`Buffer`].
    /// Allocates a gradient [`Buffer`] if it does not exist.
    #[inline]
    pub fn get_mut<'a, T, S, D>(&mut self, device: &'a D, id: Id) -> &mut Buffer<'static, T, D, S>
    where
        T: Unit + 'static,
        S: Shape,
        D: ZeroGrad<T> + Alloc<T> + 'static,
    {
        let mut new_buf = false;
        self.grads_pool
            .add_buf_once::<T, D, S>(device, id, &mut new_buf);

        if new_buf {
            self.add_zero_grad_cb::<T, D, S>(&id);
        }
        self.grads_pool.get_buf_mut(device, id).unwrap()
    }

    /// Returns a reference to a gradient [`Buffer`] using information from `buf`.
    #[inline]
    pub fn get_like<'a, T, S, D>(&mut self, buf: &Buffer<'a, T, D, S>) -> &Buffer<'static, T, D, S>
    where
        T: Unit + 'static,
        S: Shape,
        D: Alloc<T> + ZeroGrad<T> + 'static,
        D::Data<'a, T, S>: HasId,
    {
        self.get_ref(buf.device(), buf.id())
    }
}

#[cfg(test)]
mod tests {

    #[cfg(feature = "cpu")]
    #[test]
    fn test_zero_grad_on_gradients() {
        use crate::{Base, CPU, Device, Gradients, HasId};

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

        grads.zero_grad(&dev);
        let grad = grads.get_ref::<i32, (), _>(&dev, lhs.id());
        assert_eq!(grad.as_slice(), &[0; 4]);
    }
}
