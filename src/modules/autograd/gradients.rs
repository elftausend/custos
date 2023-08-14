use crate::{
    Alloc, Base, BorrowCache, Buffer, CachingError, Device, HasId, Id, Parents, Shape, CPU,
};

const INVALID_ID: &'static str = "A matching Buffer does not exist.";

/// A cache for gradients.
/// The cache is populated by `get_ref`, `get_like` or `get_mut_ref` calls.
#[derive(Default)]
pub struct Gradients {
    pub grads_pool: BorrowCache,
    pub no_grads_pool: BorrowCache,
}

impl core::fmt::Debug for Gradients {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Gradients")
            .field("cache", &self.grads_pool)
            .finish()
    }
}

type LhsRhsOut<'a, 'b, T, D, S> = (
    &'b Buffer<'a, T, D, S>,
    &'b Buffer<'a, T, D, S>,
    &'b mut Buffer<'a, T, D, S>,
    &'b mut Buffer<'a, T, D, S>,
    &'b Buffer<'a, T, D, S>,
);

impl Gradients {
    /// Clears the cache.
    #[inline]
    pub fn zero_grad(&mut self) {
        self.grads_pool.cache.clear();
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
        D: Alloc<T> + 'static,
    {
        self.grads_pool.add_or_get(device, id)
    }

    /// Returns a mutable reference to a gradient [`Buffer`].
    /// Allocates a gradient [`Buffer`] if it does not exist.
    #[inline]
    pub fn get_mut<'a, T, S, D>(&mut self, device: &'a D, id: Id) -> &mut Buffer<'a, T, D, S>
    where
        T: 'static,
        S: Shape,
        D: Alloc<T> + 'static,
    {
        self.grads_pool.add_or_get_mut(device, id)
    }

    /// Returns a reference to a gradient [`Buffer`] using information from `buf`.
    #[inline]
    pub fn get_like<'a, T, S, D>(&mut self, buf: &Buffer<'a, T, D, S>) -> &Buffer<'a, T, D, S>
    where
        T: 'static,
        S: Shape,
        D: Alloc<T> + 'static,
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
        self.no_grads_pool.get_buf::<T, D, S>(id).expect(INVALID_ID)
    }

    /// Returns the forward [`Buffer`]s lhs and and rhs, and the gradient `Buffer`s lhs_grad, rhs_grad and out_grad.
    /// Usefull for binary operations.
    #[inline]
    pub fn get_triple<'a, T, S, D>(
        &mut self,
        device: &'a D,
        (lid, rid, oid): (Id, Id, Id),
    ) -> LhsRhsOut<'a, '_, T, D, S>
    where
        T: 'static,
        S: Shape,
        D: Alloc<T> + 'static,
    {
        self.grads_pool.add_buf_once::<T, _, S>(device, rid);
        self.grads_pool.add_buf_once::<T, _, S>(device, oid);
        let lhs_grad_ptr = self.get_mut(device, lid) as *mut _;
        let lhs_grad = unsafe { &mut *lhs_grad_ptr };

        let rhs_grad_ptr = self.get_mut(device, rid) as *mut _;
        let rhs_grad = unsafe { &mut *rhs_grad_ptr };
        (
            self.get_buf_from_no_grad_pool(lid),
            self.get_buf_from_no_grad_pool(rid),
            lhs_grad,
            rhs_grad,
            self.may_get_ref(oid).unwrap(),
        )
    }

    /// Returns the forward [`Buffer`] x and the gradient `Buffer`s x_grad and out_grad.
    /// Useful for unary operations.
    ///
    #[inline]
    pub fn get_double<'a, T, IS, OS, D>(
        &mut self,
        // device: &'a D,
        parents: impl Parents<2>,
        // (xid, oid): (Id, Id),
    ) -> (
        &Buffer<'a, T, D, IS>,
        &mut Buffer<'a, T, D, IS>,
        &Buffer<'a, T, D, OS>,
    )
    where
        T: 'static,
        IS: Shape,
        OS: Shape,
        D: Alloc<T> + 'static,
    {
        let [xid, oid] = parents.ids();
        // self.grads_pool.add_buf_once::<T, _, IS>(device, oid);

        // let x_grad_ptr = self.get_mut(device, xid) as *mut _;
        let x_grad_ptr = self.may_get_mut(xid).unwrap() as *mut _;
        let x_grad_mut = unsafe { &mut *x_grad_ptr };
        let o_grad = self.may_get_ref(oid).unwrap();

        (self.get_buf_from_no_grad_pool(xid), x_grad_mut, o_grad)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Autograd, Base, Buffer, HasId, Retriever, CPU};

    #[test]
    fn test_same_types_get_double_return() {
        let device = CPU::<Autograd<Base>>::new();

        // let mut gradients = Gradients::default();

        let buf = Buffer::<i32, _>::new(&device, 10);
        // unsafe { register_buf(&mut gradients.no_grads_pool.borrow_mut().cache, &buf) }

        let out: Buffer<i32, _> = device.retrieve::<(), 0>(buf.len(), ());
        // unsafe { register_buf(&mut gradients.no_grads_pool.borrow_mut().cache, &out) }

        device
            .modules
            .tape
            .borrow_mut()
            .grads
            .get_double::<i32, (), (), CPU<Autograd<crate::CachedModule<Base, CPU<Autograd<Base>>>>>>((buf.id(), out.id()));
    }

    #[test]
    #[should_panic]
    fn test_different_types_get_double_return() {
        let device = CPU::<Autograd<Base>>::new();

        // let mut gradients = Gradients::default();

        let buf = Buffer::<i32, _>::new(&device, 10);
        // unsafe { register_buf(&mut gradients.no_grads_pool.borrow_mut().cache, &buf) }

        let out: Buffer<i64, _> = device.retrieve::<(), 0>(buf.len(), ());
        // unsafe { register_buf(&mut gradients.no_grads_pool.borrow_mut().cache, &out) }

        device
            .modules
            .tape
            .borrow_mut()
            .grads
            .get_double::<i32, (), (), CPU<Autograd<crate::CachedModule<Base, CPU<Autograd<Base>>>>>>((buf.id(), out.id()));
    }
}
