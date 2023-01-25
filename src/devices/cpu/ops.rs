use crate::{
    ApplyFunction, Buffer, ClearBuf, Device, Eval, MainMemory, Read, Resolve, Shape, ToVal,
    WriteBuf, CPU,
};

impl<T, D: MainMemory, S: Shape> Read<T, D, S> for CPU {
    type Read<'a> = &'a [T] where T: 'a, D: 'a, S: 'a;

    #[inline]
    fn read<'a>(&self, buf: &'a Buffer<T, D, S>) -> Self::Read<'a> {
        buf.as_slice()
    }

    #[inline]
    fn read_to_vec<'a>(&self, buf: &Buffer<T, D, S>) -> Vec<T>
    where
        T: Default + Clone,
    {
        buf.to_vec()
    }
}

impl<T: Default, D: MainMemory> ClearBuf<T, D> for CPU {
    fn clear(&self, buf: &mut Buffer<T, D>) {
        for value in buf {
            *value = T::default();
        }
    }
}

impl<T: Copy, D: MainMemory> WriteBuf<T, D> for CPU {
    fn write(&self, buf: &mut Buffer<T, D>, data: &[T]) {
        buf.copy_from_slice(data)
    }
}

impl<T, D, S> ApplyFunction<T, S, D> for CPU
where
    T: Copy + Default + ToVal,
    D: crate::MainMemory,
    S: Shape,
{
    fn apply_fn<F>(&self, buf: &Buffer<T, D, S>, f: impl Fn(Resolve<T>) -> F) -> Buffer<T, Self, S>
    where
        F: Eval<T>,
    {
        let mut out = self.retrieve::<T, S>(buf.len());

        for (value, x) in out.iter_mut().zip(buf.iter()) {
            *value = f((*x).to_val()).eval()
        }

        out
    }
}
