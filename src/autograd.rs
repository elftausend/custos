use core::{cell::{RefMut, Ref}, fmt::Debug, marker::PhantomData};

use crate::{prelude::One, Alloc, Buffer, Ident, RawConv, Shape, WriteBuf, Device, borrowing_cache::BorrowingCache};

#[derive(Default)]
pub struct Gradients<D> {
    // maybe use a borrowed cache in the style of the 'owned' cache
    cache: BorrowingCache,
    _pd: PhantomData<D>
}

impl<D: RawConv> Debug for Gradients<D>
where
    D::CT: Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Gradients")
            .field("cache", &self.cache)
            .finish()
    }
}

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

    #[inline]
    pub fn zero_grad(&mut self) {
        self.cache.cache.clear();
    }

    #[inline]
    pub fn may_get_ref<'a, T, S>(
        &self,
        ident: Ident,
    ) -> Option<&Buffer<'a, T, D, S>>
    where
        T: 'static,
        S: Shape,
        D: Device,
    {
        self.cache.get_buf(ident)
    }

    #[inline]
    pub fn may_get_mut<'a, T, S>(
        &mut self,
        ident: Ident,
    ) -> Option<&mut Buffer<'a, T, D, S>>
    where
        T: 'static,
        S: Shape,
        D: Device,
    {
        self.cache.get_buf_mut(ident)
    }

    #[inline]
    pub fn get_ref<'a, T, S>(
        &mut self,
        device: &'a D,
        ident: Ident,
    ) -> &Buffer<'a, T, D, S>
    where
        T: 'static,
        S: Shape,
        D: Alloc<'a, T, S>,
    {
        self.cache.add_or_get(device, ident)
    }

    #[inline]
    pub fn get_mut<'a, T, S>(
        &mut self,
        device: &'a D,
        ident: Ident,
    ) -> &mut Buffer<'a, T, D, S>
    where
        T: 'static,
        S: Shape,
        D: for<'b> Alloc<'b, T, S>,
    {
       self.cache.add_or_get_mut(device, ident)
    }

    #[inline]
    pub fn get_like<'a, T, S>(&mut self, buf: &Buffer<'a, T, D, S>) -> &Buffer<'a, T, D, S>
    where
        T: 'static,
        S: Shape,
        D: Alloc<'a, T, S>,
    {
        self.get_ref(buf.device(), buf.id())
    }

    #[inline]
    pub fn get_triple<'a, T, S>(
        &mut self,
        device: &'a D,
        (lid, rid, oid): (Ident, Ident, Ident),
    ) -> (
        Buffer<'a, T, D, S>,
        Buffer<'a, T, D, S>,
        &mut Buffer<'a, T, D, S>,
        &mut Buffer<'a, T, D, S>,
        &Buffer<'a, T, D, S>,
    ) 
    where
        T: 'static,
        S: Shape,
        D: for<'b> Alloc<'b, T, S>,
    {
        self.cache.add_buf_once(device, rid);
        self.cache.add_buf_once(device, oid);

        let lhs_grad_ptr = self.get_mut(device, lid) as *mut _;
        let lhs_grad = unsafe { &mut *lhs_grad_ptr };

        let rhs_grad_ptr = self.get_mut(device, rid) as *mut _;
        let rhs_grad = unsafe { &mut *rhs_grad_ptr };
        (
            device.get_existing_buf(lid),
            device.get_existing_buf(rid),
            lhs_grad,    
            rhs_grad,
            self.may_get_ref(oid).unwrap()
        )
    }

    #[inline]
    pub fn get_double<'a, T, S>(
        &mut self,
        device: &'a D,
        (xid, oid): (Ident, Ident),
    ) -> (
        Buffer<'a, T, D, S>,
        &mut Buffer<'a, T, D, S>,
        &Buffer<'a, T, D, S>,
    )
    where
        T: 'static,
        S: Shape,
        D: for<'b> Alloc<'b, T, S>,
    {
        let x_grad_ptr = self.get_mut(device, xid) as *mut _;
        let x_grad_mut = unsafe { &mut *x_grad_ptr };
        let o_grad = self.get_ref(&device, oid);

        (
            device.get_existing_buf(xid),
            x_grad_mut,
            o_grad,
        )
    }
}

#[derive(Default)]
pub struct Tape<D: RawConv> {
    pub grads: Gradients<D>,
    grad_fns: Vec<Box<dyn Fn(&mut Gradients<D>, &D)>>,
}

pub trait TapeReturn: RawConv {
    fn tape(&self) -> Ref<Tape<Self>>;
    fn tape_mut(&self) -> RefMut<Tape<Self>>;
}

impl<D: RawConv> Debug for Tape<D>
where
    D::CT: Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Tape").field("grads", &self.grads).finish()
    }
}

impl<D: RawConv> Tape<D> {
    #[inline]
    pub fn add_grad_fn<F: Fn(&mut Gradients<D>, &D) + 'static>(&mut self, grad_fn: F) {
        self.grad_fns.push(Box::new(grad_fn))
    }

    pub fn backward(&mut self, device: &D) {
        for grad_fn in self.grad_fns.drain(..).rev() {
            grad_fn(&mut self.grads, device);
        }
    }

    pub fn backward_seeded<T, S: Shape>(&mut self, buf: &Buffer<T, D, S>)
    where
        T: Clone + One + 'static,
        D: for<'a> Alloc<'a, T, S> + WriteBuf<T, S, D>,
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
    D: TapeReturn + WriteBuf<T, S, D> + for<'b> Alloc<'b, T, S>,
    S: Shape,
{
    #[inline]
    pub fn backward(&self) {
        self.device().tape_mut().backward_seeded(self)
    }
    
    #[inline]
    pub fn grad(&self) -> Ref<Self> {
        Ref::map(self.device().tape(), |tape| {
            tape.grads.may_get_ref(self.id()).unwrap()
        })
    }

    #[inline]
    pub fn grad_unbound(&self) -> Ref<'a, Self> {
        Ref::map(self.device().tape(), |tape| {
            tape.grads.may_get_ref(self.id()).unwrap()
        })
    }

    #[inline]
    pub fn grad_mut(&mut self) -> RefMut<Self> {
        RefMut::map(self.device().tape_mut(), |tape| {
            tape.grads.may_get_mut(self.id()).unwrap()
        })
    }

    #[inline]
    pub fn grad_mut_unbound(&mut self) -> RefMut<'a, Self> {
        RefMut::map(self.device().tape_mut(), |tape| {
            tape.grads.may_get_mut(self.id()).unwrap()
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{Buffer, Combiner};

    #[cfg(feature = "cpu")]
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
        use crate::{OpenCL, UnaryElementWiseMayGrad};

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
