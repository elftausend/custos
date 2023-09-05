use core::{
    any::Any,
    ops::{Index, Range, RangeBounds},
};

use crate::{
    bounds_to_range, AddOperation, Alloc, Buffer, ClearBuf, CopySlice, Device, HasId, MainMemory,
    OnDropBuffer, Operation, Read, Retriever, Shape, WriteBuf, CPU, MayTapeActions,
};

impl<Mods, T, S, D> crate::ApplyFunctionLazyTest<T, S, D> for CPU<Mods>
where
    Mods: crate::Retrieve<Self, T> + MayTapeActions + AddOperation + 'static,
    T: Copy + Default + crate::ToVal + 'static,
    S: Shape,
    D: MainMemory + Alloc<T> + 'static,
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

        #[cfg(feature = "autograd")]
        self.add_grad_fn(move |grads| {
            let (lhs, lhs_grad, out_grad) = grads.get_double::<T, S, S, D>(ids);
        });

        unsafe {
            self.add_operation(&mut out, move |out| {
                let out = out.downcast_mut::<Buffer<T, D, S>>().unwrap();

                for (x, out) in buf.iter().zip(out.iter_mut()) {
                    *out = f((*x).to_val()).eval();
                }
            });
        }
        out
    }
}

impl<Mods: AddOperation> AddOperation for CPU<Mods> {
    #[inline]
    unsafe fn add_operation<T: 'static, D: Device + 'static, S: Shape>(
        &self,
        out: &mut Buffer<T, D, S>,
        operation: impl Fn(&mut dyn Any),
    ) {
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

impl<Mods: OnDropBuffer, T, D: MainMemory, S: Shape> Read<T, S, D> for CPU<Mods> {
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

// #[impl_stack]
impl<Mods: OnDropBuffer, T: Default, D: MainMemory, S: Shape> ClearBuf<T, S, D> for CPU<Mods> {
    fn clear(&self, buf: &mut Buffer<T, D, S>) {
        for value in buf {
            *value = T::default();
        }
    }
}

impl<Mods: OnDropBuffer, T: Copy, D: MainMemory> CopySlice<T, D> for CPU<Mods>
where
    [T]: Index<Range<usize>, Output = [T]>,
{
    fn copy_slice_to<SR: RangeBounds<usize>, DR: RangeBounds<usize>>(
        &self,
        source: &Buffer<T, D>,
        source_range: SR,
        dest: &mut Buffer<T, Self>,
        dest_range: DR,
    ) {
        let source_range = bounds_to_range(source_range, source.len());
        let dest_range = bounds_to_range(dest_range, dest.len());

        assert_eq!(
            source_range.end - source_range.start,
            dest_range.end - dest_range.start,
        );

        dest[dest_range].copy_from_slice(&source[source_range]);
    }

    fn copy_slice_all<I: IntoIterator<Item = (Range<usize>, Range<usize>)>>(
        &self,
        source: &Buffer<T, D>,
        dest: &mut Buffer<T, Self>,
        ranges: I,
    ) {
        for (source_range, dest_range) in ranges {
            self.copy_slice_to(source, source_range, dest, dest_range);
        }
    }
}
