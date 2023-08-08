use crate::{
    module_comb::{
        ApplyFunction, Buffer, Device, HasId, MainMemory, OnDropBuffer, Retrieve, Retriever,
        TapeActions, WriteBuf,
    },
    Shape,
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

impl<Mods: Retrieve<Self> + TapeActions + 'static, T: 'static, S: Shape, D: MainMemory>
    ApplyFunction<T, S, D> for CPU<Mods>
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
        let out = self.retrieve(buf.len());

        let ids = (buf.id(), out.id());
        self.add_grad_fn::<T, S, 2>(ids, move |grads| {
            // let (lhs, lhs_grad, out_grad) = grads.get_double::<T, S, S>(ids);
        });

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
