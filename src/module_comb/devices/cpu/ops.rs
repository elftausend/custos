use core::any::Any;

use crate::{
    module_comb::{
        AddOperation, ApplyFunction, Buffer, Device, HasId, MainMemory, OnDropBuffer, Retrieve,
        Retriever, TapeActions, WriteBuf, Operation,
    },
    Eval, Shape, ToVal,
};

use super::CPU;

impl<Mods: OnDropBuffer, T: Copy, D: MainMemory, S: Shape> WriteBuf<T, S, D> for CPU<Mods> {
    #[inline]
    fn write(&self, buf: &mut Buffer<T, D, S>, data: &[T]) {
        buf.copy_from_slice(data)
    }

    #[inline]
    fn write_buf(&self, dst: &mut Buffer<T, D, S>, src: &Buffer<T, D, S>) {
        self.write(dst, src)
    }
}

impl<Mods: AddOperation> AddOperation for CPU<Mods> {
    #[inline]
    unsafe fn add_operation<T: 'static, D: Device + 'static, S: Shape>(&self, out: &mut Buffer<T, D, S>, operation: impl Fn(&mut dyn Any)) {
        self.modules.add_operation(out, operation)
    }

    #[inline]
    fn add_operation2(&self, operation: impl Operation) {
        self.modules.add_operation2(operation)
    }

    #[inline]
    fn call_lazily(&self) {
        self.modules.call_lazily()
    }
}

impl<Mods, T, S, D> ApplyFunction<T, S, D> for CPU<Mods>
where
    Mods: Retrieve<Self> + TapeActions + AddOperation + 'static,
    T: Copy + Default + ToVal + 'static,
    S: Shape,
    D: MainMemory + 'static,
{
    // actually take &mut Buf instead of returning an owned Buf?
    fn apply_fn<F>(
        &self,
        buf: &Buffer<T, D, S>,
        f: impl Fn(crate::Resolve<T>) -> F + Copy, // without Copy -> UB haha
    ) -> Buffer<T, Self, S>
    where
        F: crate::Eval<T> + crate::MayToCLSource,
    {
        let mut out = self.retrieve(buf.len(), buf);

        let ids = (buf.id(), out.id());
        self.add_grad_fn::<T, S>(move |grads| {
            let (lhs, lhs_grad, out_grad) = grads.get_double::<T, S, S, D>(ids);
        });

        unsafe {
            self.add_operation(&mut out, move |out| {
                let out = out.downcast_mut::<Buffer<T, D, S>>()
                    .unwrap();

                for (x, out) in buf.iter().zip(out.iter_mut()) {
                    *out = f((*x).to_val()).eval();
                }
                    
            });
        }
        // self.add_operation2(ApplyFn { buf, out: out.clone(), f });
        // apply_fn_slice(buf, &mut out, f);

        out
    }
}


pub struct ApplyFn<'b, 'a, 'c, T, D: Device, S: Shape, D1: Device, S1: Shape, O: Eval<T>, F: Fn(crate::Resolve<T>) -> O> {
    buf: &'b Buffer<'a, T, D, S>,
    out: Buffer<'c, T, D1, S1>,
    f: F
}

impl<'a, 'b,  'c, T: Copy, D: MainMemory, S: Shape, D1: MainMemory, S1: Shape, O: Eval<T>, F: Fn(crate::Resolve<T>) -> O> Operation for ApplyFn<'b, 'a, 'c, T, D, S, D1, S1, O, F> {
    fn forward(&mut self) {

        apply_fn_slice(self.buf, &mut self.out, &self.f);
    }
}

fn apply_fn_slice<T, O>(
    x: &[T],
    out: &mut [T],
    f: impl Fn(crate::Resolve<T>) -> O,
) where
    T: Copy,
    O: Eval<T>,
{
    for (x, out) in x.iter().zip(out.iter_mut()) {
        *out = f((*x).to_val()).eval();
    }
}

#[cfg(test)]
mod tests {
    use crate::module_comb::{Base, Buffer, Cached, WriteBuf, CPU};

    #[test]
    fn test_same_core_device_different_modules() {
        let dev1 = CPU::<Base>::new();
        let dev2 = CPU::<Cached<Base>>::new();

        let mut buf_from_dev2 = Buffer::<_, _, ()>::new(&dev2, 10);
        dev1.write(&mut buf_from_dev2, &[1, 2, 3, 4])
    }
}
