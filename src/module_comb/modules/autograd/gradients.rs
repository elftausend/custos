use core::cell::RefCell;

use crate::{
    module_comb::{Alloc, BorrowCache, Buffer, HasId, Id},
    Shape,
};

/// A cache for gradients.
/// The cache is populated by `get_ref`, `get_like` or `get_mut_ref` calls.
#[derive(Default)]
pub struct Gradients {
    pub cache: BorrowCache,
    pub no_grads_pool: RefCell<BorrowCache>,
}

impl core::fmt::Debug for Gradients {
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

impl Gradients {
    /// Clears the cache.
    #[inline]
    pub fn zero_grad(&mut self) {
        self.cache.cache.clear();
    }

    /// May get a reference to a gradient [`Buffer`].
    #[inline]
    pub fn may_get_ref<'a, T, S, D>(&self, ident: Id) -> Option<&Buffer<'a, T, D, S>>
    where
        T: 'static,
        S: Shape,
        D: Alloc + 'static,
    {
        self.cache.get_buf(ident)
    }

    /// May get a mutable reference to a gradient [`Buffer`].
    #[inline]
    pub fn may_get_mut<'a, T, S, D>(&mut self, id: Id) -> Option<&mut Buffer<'a, T, D, S>>
    where
        T: 'static,
        S: Shape,
        D: Alloc + 'static,
    {
        self.cache.get_buf_mut(id)
    }

    /// Returns a reference to a gradient [`Buffer`].
    /// Allocates a gradient [`Buffer`] if it does not exist.
    #[inline]
    pub fn get_ref<'a, T, S, D>(&mut self, device: &'a D, id: Id) -> &Buffer<'a, T, D, S>
    where
        T: 'static,
        S: Shape,
        D: Alloc + 'static,
    {
        self.cache.add_or_get(device, id)
    }

    /// Returns a mutable reference to a gradient [`Buffer`].
    /// Allocates a gradient [`Buffer`] if it does not exist.
    #[inline]
    pub fn get_mut<'a, T, S, D>(&mut self, device: &'a D, id: Id) -> &mut Buffer<'a, T, D, S>
    where
        T: 'static,
        S: Shape,
        D: Alloc + 'static,
    {
        self.cache.add_or_get_mut(device, id)
    }

    /// Returns a reference to a gradient [`Buffer`] using information from `buf`.
    #[inline]
    pub fn get_like<'a, T, S, D>(&mut self, buf: &Buffer<'a, T, D, S>) -> &Buffer<'a, T, D, S>
    where
        T: 'static,
        S: Shape,
        D: Alloc + 'static,
        D::Data<T, S>: HasId,
    {
        self.get_ref(buf.device(), buf.id())
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
        D: Alloc + 'static,
    {
        todo!("works fine -> no_grads_pool");

        /*
        self.cache.add_buf_once(device, rid);
        self.cache.add_buf_once(device, oid);
        let lhs_grad_ptr = self.get_mut(device, lid) as *mut _;
        let lhs_grad = unsafe { &mut *lhs_grad_ptr };

        let rhs_grad_ptr = self.get_mut(device, rid) as *mut _;
        let rhs_grad = unsafe { &mut *rhs_grad_ptr };
        (
            // self.no_grads_pool.get_buf(lid)
            unsafe { device.get_existing_buf(lid) },
            unsafe { device.get_existing_buf(rid) },
            lhs_grad,
            rhs_grad,
            self.may_get_ref(oid).unwrap(),
        )*/
    }

    /// Returns the forward [`Buffer`] x and the gradient `Buffer`s x_grad and out_grad.
    /// Usefull for unary operations.
    #[inline]
    pub fn get_double<'a, T, IS, OS, D>(
        &mut self,
        device: &'a D,
        (xid, oid): (Id, Id),
    ) -> (
        Buffer<'a, T, D, IS>,
        &mut Buffer<'a, T, D, IS>,
        &Buffer<'a, T, D, OS>,
    )
    where
        T: 'static,
        IS: Shape,
        OS: Shape,
        D: Alloc + 'static,
    {
        todo!("works fine -> no_grads_pool");
        /*let x_grad_ptr = self.get_mut(device, xid) as *mut _;
        let x_grad_mut = unsafe { &mut *x_grad_ptr };
        let o_grad = self.get_ref(device, oid);

        (unsafe { device.get_existing_buf(xid) }, x_grad_mut, o_grad)*/
    }
}
