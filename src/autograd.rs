use core::{cell::RefMut, fmt::Debug};

use crate::{
    prelude::One, Alloc, Buffer, Cache, Ident, RawConv,
    Shape, WriteBuf,
};

#[derive(Default)]
pub struct Gradients<D: RawConv> {
    // Borrowing cache
    cache: Cache<D>,
}

impl<D: RawConv> Debug for Gradients<D>
where
    D::CT: Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Gradients")
            .field("cache", &self.cache.nodes)
            .finish()
    }
}

impl<D: RawConv> Gradients<D> {
    #[inline]
    pub fn get_like_raw<'a, T, S: Shape>(
        &mut self,
        device: &'a D,
        ident: Ident,
    ) -> Buffer<'a, T, D, S>
    where
        D: Alloc<'a, T, S>,
    {
        self.cache.get(device, ident, || ())
    }

    #[inline]
    pub fn get_like<'a, T, S: Shape>(&mut self, buf: &Buffer<'a, T, D, S>) -> Buffer<'a, T, D, S>
    where
        D: Alloc<'a, T, S>,
    {
        self.get_like_raw(buf.device(), buf.id())
    }

    #[inline]
    pub fn get_triple<'a, T, S: Shape>(
        &mut self,
        device: &'a D,
        (lid, rid, oid): (Ident, Ident, Ident),
    ) -> (
        Buffer<'a, T, D, S>,
        Buffer<'a, T, D, S>,
        Buffer<'a, T, D, S>,
        Buffer<'a, T, D, S>,
        Buffer<'a, T, D, S>,
    )
    where
        D: for<'b> Alloc<'b, T, S>,
    {
        (
            device.get_existing_buf(lid),
            device.get_existing_buf(rid),
            self.get_like_raw(device, lid),
            self.get_like_raw(device, rid),
            self.get_like_raw(device, oid),
        )
    }

    #[inline]
    pub fn get_double<'a, T, S: Shape>(
        &mut self,
        device: &'a D,
        (xid, oid): (Ident, Ident),
    ) -> (
        Buffer<'a, T, D, S>,
        Buffer<'a, T, D, S>,
        Buffer<'a, T, D, S>,
    )
    where
        D: for<'b> Alloc<'b, T, S>,
    {
        (
            device.get_existing_buf(xid),
            self.get_like_raw(device, xid),
            self.get_like_raw(device, oid),
        )
    }
}

#[derive(Default)]
pub struct Tape<D: RawConv> {
    grads: Gradients<D>,
    grad_fns: Vec<Box<dyn Fn(&mut Gradients<D>, &D)>>,
}

pub trait TapeReturn: RawConv {
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
        T: Clone + One,
        D: for<'a> Alloc<'a, T, S> + WriteBuf<T, S, D>,
    {
        let mut out = self.grads.get_like::<T, S>(buf);
        out.write(&vec![T::one(); out.len()]);

        self.backward(buf.device())
    }
}

impl<'a, T, D, S> Buffer<'a, T, D, S>
where
    T: Clone + One,
    D: TapeReturn + WriteBuf<T, S, D> + for<'b> Alloc<'b, T, S>,
    S: Shape,
{
    #[inline]
    pub fn backward(&self) {
        self.device().tape_mut().backward_seeded(self)
    }

    #[inline]
    pub fn grad(&self) -> Self {
        self.device().tape_mut().grads.get_like(&self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Buffer, Combiner};


    #[cfg(feature = "cpu")]
    #[test]
    fn test_tape_unary_ew() {
        use crate::{CPU, UnaryElementWiseMayGrad};

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
