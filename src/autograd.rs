use core::{cell::RefMut, fmt::Debug};

use crate::{
    prelude::One, Alloc, ApplyFunction, Buffer, Cache, Device, Eval, Ident, RawConv, Resolve,
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
            device.get_like(lid),
            device.get_like(rid),
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
            device.get_like(xid),
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
        D: for<'a> Alloc<'a, T, S> + WriteBuf<T, D, S>,
    {
        let mut out = self.grads.get_like::<T, S>(buf);
        out.write(&vec![T::one(); out.len()]);

        self.backward(buf.device())
    }
}

pub trait UnaryElementWise<T, D: Device, S: Shape>: Device {
    fn unary_ew<FO, GO>(
        &self,
        buf: &Buffer<T, D, S>,
        forward_fn: impl Fn(Resolve<T>) -> FO,
        x_grad: fn(&T) -> GO,
    ) -> Buffer<T, Self, S>
    where
        FO: Eval<T> + ToString,
        GO: Eval<T> + ToString;
}

impl<T, D, S> UnaryElementWise<T, D, S> for D
where
    D: ApplyFunction<T, S, D> + TapeReturn,
    D: for<'a> Alloc<'a, T, S>,
    S: Shape,
{
    fn unary_ew<FO, GO>(
        &self,
        buf: &Buffer<T, D, S>,
        forward_fn: impl Fn(Resolve<T>) -> FO,
        x_grad: fn(&T) -> GO,
    ) -> Buffer<T, Self, S>
    where
        FO: Eval<T> + ToString,
        GO: Eval<T> + ToString,
    {
        let out = self.apply_fn(buf, forward_fn);

        let ids = (buf.id(), out.id());
        self.tape_mut().add_grad_fn(move |grads, device| {
            let (lhs, mut lhs_grad, out_grad) = grads.get_double::<T, S>(device, ids);

            //unary_grad_slice(lhs.len(), &lhs, &mut lhs_grad, &out_grad, x_grad);
        });

        out
    }
}

#[cfg(test)]
mod tests {
    use crate::{Buffer, Tape, UnaryElementWise, CPU};

    #[test]
    fn test_tape_unary_ew() {
        let device = CPU::new();

        let buf = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

        device.unary_ew(&buf, |x| x, |x| *x);
    }
}
