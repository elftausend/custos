use crate::{
    module_comb::{
        AddOperation, ApplyFunction, Buffer, HasId, MainMemory, OnDropBuffer, Retrieve, Retriever,
        TapeActions, WriteBuf,
    },
    Shape, ToVal,
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
    fn add_operation(&self, operation: impl FnOnce()) {
        self.modules.add_operation(operation)
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
        f: impl Fn(crate::Resolve<T>) -> F,
    ) -> Buffer<T, Self, S>
    where
        F: crate::Eval<T> + crate::MayToCLSource,
    {
        let mut out = self.retrieve(buf.len(), buf);

        let ids = (buf.id(), out.id());
        self.add_grad_fn::<T, S>(move |grads| {
            let (lhs, lhs_grad, out_grad) = grads.get_double::<T, S, S, D>(ids);
        });

        for (x, out) in buf.iter().zip(out.iter_mut()) {
            *out = f((*x).to_val()).eval();
        }

        out
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
