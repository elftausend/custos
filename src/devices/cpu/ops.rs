use core::{
    fmt::Debug,
    ops::{AddAssign, Deref, DerefMut, Index, Range, RangeBounds},
};

use crate::{
    bounds_to_range,
    cpu_stack_ops::{apply_fn_slice, clear_slice},
    pass_down_add_operation, pass_down_exec_now, AddOperation, ApplyFunction, AsNoId, BufAsNoId,
    Buffer, ClearBuf, CopySlice, Device, Eval, MayToCLSource, OnDropBuffer, Read, Resolve,
    Retrieve, Retriever, Shape, ToVal, UnaryGrad, WriteBuf, ZeroGrad, CPU,
};

pass_down_add_operation!(CPU);
pass_down_exec_now!(CPU);

impl<Mods, T, D, S> ApplyFunction<T, S, D> for CPU<Mods>
where
    Mods: Retrieve<Self, T, S> + AddOperation + 'static,
    T: Copy + Default + ToVal + 'static,
    D: Device + 'static,
    D::Base<T, S>: Deref<Target = [T]>,
    S: Shape,
{
    fn apply_fn<F>(
        &self,
        buf: &Buffer<T, D, S>,
        f: impl Fn(Resolve<T>) -> F + Copy + 'static,
    ) -> Buffer<T, Self, S>
    where
        F: Eval<T> + MayToCLSource,
    {
        let mut out = self.retrieve(buf.len(), buf);

        self.add_op((&mut out, buf, f.no_id()), move |(out, buf, f)| {
            apply_fn_slice(buf, out, **f);
            Ok(())
        })
        .unwrap();

        // self.add_op((buf, f.no_id()), Some(&mut out), move |out, (buf, f)| {
        //     apply_fn_slice(buf, out.as_mut().unwrap(), **f);
        //     Ok(())
        // })
        // .unwrap();

        out
    }
}

impl<Mods, T, D, S> UnaryGrad<T, S, D> for CPU<Mods>
where
    Mods: AddOperation + OnDropBuffer,
    T: AddAssign + Copy + std::ops::Mul<Output = T> + 'static,
    S: Shape,
    D: Device + 'static,
    D::Base<T, S>: Deref<Target = [T]> + DerefMut<Target = [T]>,
{
    #[inline]
    fn add_unary_grad<F>(
        &self,
        lhs: &Buffer<T, D, S>,
        lhs_grad: &mut Buffer<T, D, S>,
        out: &Buffer<T, D, S>,
        lhs_grad_fn: impl Fn(Resolve<T>) -> F + Copy + 'static,
    ) where
        F: Eval<T> + MayToCLSource,
    {
        self.add_op::<_, 4>(
            (lhs, lhs_grad.buf_no_id(), out, lhs_grad_fn.no_id()),
            |(lhs, lhs_grad, out, lhs_grad_fn)| {
                crate::cpu_stack_ops::add_unary_grad(lhs, out, lhs_grad, **lhs_grad_fn);
                Ok(())
            },
        )
        .unwrap();
    }
}

impl<Mods, T, D, S> Read<T, S, D> for CPU<Mods>
where
    Mods: OnDropBuffer,
    D: Device,
    D::Base<T, S>: Deref<Target = [T]>,
    S: Shape,
{
    type Read<'a> = &'a [T] where T: 'a, D: 'a, S: 'a;

    #[inline]
    fn read<'a>(&self, buf: &'a Buffer<T, D, S>) -> Self::Read<'a> {
        &**buf
    }

    #[inline]
    fn read_to_vec<'a>(&self, buf: &Buffer<T, D, S>) -> Vec<T>
    where
        T: Default + Clone,
    {
        buf.to_vec()
    }
}

impl<Mods, T, D, S> WriteBuf<T, S, D> for CPU<Mods>
where
    Mods: OnDropBuffer,
    T: Copy,
    D: Device,
    D::Base<T, S>: DerefMut<Target = [T]>,
    S: Shape,
{
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
impl<Mods, T, D, S> ClearBuf<T, S, D> for CPU<Mods>
where
    Mods: OnDropBuffer + AddOperation,
    T: Default + 'static,
    D: Device + 'static,
    D::Base<T, S>: DerefMut<Target = [T]>,
    S: Shape,
{
    #[inline]
    fn clear(&self, buf: &mut Buffer<T, D, S>) {
        clear_slice(buf);
    }
}

impl<Mods, T> ZeroGrad<T> for CPU<Mods>
where
    T: Default,
    Mods: OnDropBuffer,
{
    #[inline]
    fn zero_grad<S: Shape>(&self, data: &mut Self::Base<T, S>) {
        clear_slice(data)
    }
}

impl<Mods, T, D> CopySlice<T, D> for CPU<Mods>
where
    [T]: Index<Range<usize>, Output = [T]>,
    Mods: OnDropBuffer,
    T: Copy,
    D: Device,
    D::Base<T, ()>: Deref<Target = [T]>,
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
