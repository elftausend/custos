use core::ops::{AddAssign, Index, Range, RangeBounds};

use crate::{
    bounds_to_range, cpu_stack_ops::clear_slice, AddOperation, Alloc, ApplyFunction, Buffer,
    ClearBuf, CopySlice, Device, Eval, HasId, MainMemory, MayTapeActions, MayToCLSource,
    OnDropBuffer, Read, Resolve, Retrieve, Retriever, Shape, ToVal, UnaryGrad, WriteBuf, CPU, pass_down_add_operation,
};

#[cfg(feature = "autograd")]
use crate::TapeActions;

impl<Mods, T, S, D> crate::ApplyFunctionLazyTest<T, S, D> for CPU<Mods>
where
    Mods: crate::Retrieve<Self, T> + MayTapeActions + AddOperation<T, Self> + 'static,
    T: Copy + Default + crate::ToVal + 'static,
    S: Shape,
    D: MainMemory + Alloc<T> + 'static,
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

        #[cfg(feature = "autograd")]
        self.add_grad_fn(move |grads| {
            let (_lhs, _lhs_grad, _out_grad) = grads.get_double::<T, S, S, D>(ids);
        });

        self.add_op(&mut out, |out| {
            for (x, out) in buf.iter().zip(out.iter_mut()) {
                *out = f((*x).to_val()).eval();
            }
        });

        out
    }
}

pass_down_add_operation!(CPU);

impl<Mods, T, D, S> ApplyFunction<T, S, D> for CPU<Mods>
where
    Mods: Retrieve<Self, T>,
    T: Copy + Default + ToVal + 'static,
    D: crate::MainMemory,
    S: Shape,
{
    fn apply_fn<F>(&self, buf: &Buffer<T, D, S>, f: impl Fn(Resolve<T>) -> F) -> Buffer<T, Self, S>
    where
        F: Eval<T> + MayToCLSource,
    {
        let mut out = self.retrieve(buf.len(), buf);

        crate::cpu_stack_ops::apply_fn_slice(buf, &mut out, f);

        out
    }
}

impl<Mods, T, D, S> UnaryGrad<T, S, D> for CPU<Mods>
where
    Mods: OnDropBuffer,
    T: AddAssign + Copy + std::ops::Mul<Output = T>,
    S: Shape,
    D: MainMemory,
{
    #[inline]
    fn add_unary_grad<F>(
        &self,
        lhs: &Buffer<T, D, S>,
        lhs_grad: &mut Buffer<T, D, S>,
        out: &Buffer<T, D, S>,
        lhs_grad_fn: impl Fn(Resolve<T>) -> F,
    ) where
        F: Eval<T> + MayToCLSource,
    {
        crate::cpu_stack_ops::add_unary_grad(lhs, out, lhs_grad, lhs_grad_fn)
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
    #[inline]
    fn clear(&self, buf: &mut Buffer<T, D, S>) {
        clear_slice(buf)
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
