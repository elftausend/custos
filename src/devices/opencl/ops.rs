use core::ops::{Range, RangeBounds};

use min_cl::{api::{
    enqueue_copy_buffer, enqueue_copy_buffers, enqueue_full_copy_buffer, enqueue_read_buffer,
    enqueue_write_buffer, wait_for_event,
}, CLDevice};

use crate::{
    bounds_to_range, cpu_stack_ops::clear_slice, pass_down_add_operation, pass_down_exec_now,
    prelude::Number, AddOperation, ApplyFunction, Buffer, CDatatype, ClearBuf, CopySlice, Eval,
    OnDropBuffer, OpenCL, Read, Resolve, Retrieve, Retriever, Shape, ToCLSource, ToMarker,
    UnaryGrad, UseGpuOrCpu, WriteBuf,
};

use super::{enqueue_kernel, CLPtr};

/*impl<Mods: OnDropBuffer, T: CDatatype> ClearBuf<T> for OpenCL<Mods> {
    #[inline]
    fn clear(&self, buf: &mut Buffer<T, OpenCL<Mods>>) {
        try_cl_clear(self, buf).unwrap()
    }
}*/

pass_down_add_operation!(OpenCL);
pass_down_exec_now!(OpenCL);

impl<Mods: OnDropBuffer + UseGpuOrCpu, T: CDatatype + Default> ClearBuf<T> for OpenCL<Mods> {
    #[inline]
    fn clear(&self, buf: &mut Buffer<T, OpenCL<Mods>>) {
        /*crate::fork!(
        self,
        || clear_slice(buf),
        || try_cl_clear(self, buf).unwrap(),
        &[buf.len()] // TODO: (macro) could go through the params of clear_slice and add to list if buffer
        );*/
        #[cfg(unified_cl)]
        {
            let cpu_buf = unsafe { &mut *(buf as *mut Buffer<_, OpenCL<Mods>, _>) };
            self.use_cpu_or_gpu(
                (file!(), line!(), column!()).into(),
                &[buf.len()],
                || clear_slice(cpu_buf),
                || try_cl_clear(self, buf).unwrap(),
            );
        }

        #[cfg(not(unified_cl))]
        try_cl_clear(self, buf).unwrap()
    }
}
/// Sets the elements of an OpenCL Buffer to zero.
/// # Example
/// ```
/// use custos::{OpenCL, Buffer, Read, opencl::try_cl_clear, Base};
///
/// fn main() -> Result<(), custos::Error> {
///     let device = OpenCL::<Base>::new(0)?;
///     let mut lhs = Buffer::<i16, _>::from((&device, [15, 30, 21, 5, 8]));
///     assert_eq!(device.read(&lhs), vec![15, 30, 21, 5, 8]);
///
///     try_cl_clear(&device, &mut lhs)?;
///     assert_eq!(device.read(&lhs), vec![0; 5]);
///     Ok(())
/// }
/// ```
pub fn try_cl_clear<T: CDatatype>(
    device: &CLDevice,
    lhs: &mut CLPtr<T>
) -> crate::Result<()> {
    let src = format!(
        "
        __kernel void clear(__global {datatype}* self, long len) {{
            size_t id = get_global_id(0);
            if (id >= len) {{
                return;
            }}
            self[id] = 0;
        }}
    ",
        datatype = T::C_DTYPE_STR
    );

    let gws = [(lhs.len() / 32 + 1) * 32, 0, 0];
    enqueue_kernel(device, &src, gws, Some([32, 0, 0]), &[lhs, &lhs.len()])?;
    Ok(())
}

impl<T, S: Shape, Mods: OnDropBuffer> WriteBuf<T, S> for OpenCL<Mods> {
    #[inline]
    fn write(&self, buf: &mut Buffer<T, Self, S>, data: &[T]) {
        let event =
            unsafe { enqueue_write_buffer(self.queue(), buf.cl_ptr(), data, true).unwrap() };
        wait_for_event(event).unwrap();
    }

    #[inline]
    fn write_buf(&self, dst: &mut Buffer<T, Self, S>, src: &Buffer<T, Self, S>) {
        debug_assert_eq!(dst.len(), src.len());
        enqueue_full_copy_buffer::<T>(self.queue(), src.cl_ptr(), dst.cl_ptr(), dst.len()).unwrap();
    }
}

impl<T> CopySlice<T> for OpenCL {
    fn copy_slice_to<SR: RangeBounds<usize>, DR: RangeBounds<usize>>(
        &self,
        source: &Buffer<T, Self>,
        source_range: SR,
        dest: &mut Buffer<T, Self>,
        dest_range: DR,
    ) {
        let source_range = bounds_to_range(source_range, source.len());
        let dest_range = bounds_to_range(dest_range, dest.len());

        assert_eq!(
            source_range.end - source_range.start,
            dest_range.end - dest_range.start
        );

        enqueue_copy_buffer::<T>(
            self.queue(),
            source.data.ptr,
            dest.data.ptr,
            source_range.start,
            dest_range.start,
            source_range.end - source_range.start,
        )
        .unwrap();
    }

    fn copy_slice_all<I: IntoIterator<Item = (Range<usize>, Range<usize>)>>(
        &self,
        source: &Buffer<T, Self>,
        dest: &mut Buffer<T, Self>,
        ranges: I,
    ) {
        let ranges = ranges.into_iter().map(|(from, to)| {
            let len = from.end - from.start;
            assert_eq!(len, to.end - to.start);
            (from.start, to.start, len)
        });

        enqueue_copy_buffers::<T, _>(self.queue(), source.data.ptr, dest.data.ptr, ranges).unwrap();
    }
}

impl<Mods: OnDropBuffer + 'static, T: Clone + Default, S: Shape> Read<T, S> for OpenCL<Mods> {
    #[cfg(not(unified_cl))]
    type Read<'a> = Vec<T> where T: 'a;
    #[cfg(unified_cl)]
    type Read<'a> = &'a [T] where T: 'a;

    #[cfg(not(unified_cl))]
    fn read<'a>(&self, buf: &'a Buffer<T, OpenCL<Mods>, S>) -> Self::Read<'a> {
        self.read_to_vec(buf)
    }

    #[cfg(unified_cl)]
    #[inline]
    fn read<'a>(&self, buf: &'a Buffer<T, Self, S>) -> Self::Read<'a> {
        use crate::HostPtr;

        buf.as_slice()
    }

    #[inline]
    fn read_to_vec(&self, buf: &Buffer<T, Self, S>) -> Vec<T> {
        try_read_cl_buf_to_vec(self, buf).unwrap()
    }
}

fn try_read_cl_buf_to_vec<T: Clone + Default>(
    device: &CLDevice,
    buf: &CLPtr<T>,
) -> crate::Result<Vec<T>> {
    let mut read = vec![T::default(); buf.len()];
    let event = unsafe { enqueue_read_buffer(device.queue(), buf.ptr, &mut read, false)? };
    wait_for_event(event).unwrap();
    Ok(read)
}

impl<T, S, Mods> ApplyFunction<T, S> for OpenCL<Mods>
where
    T: CDatatype + Number,
    S: Shape,
    Mods: AddOperation<T, Self> + Retrieve<Self, T> + UseGpuOrCpu + 'static,
{
    #[inline]
    fn apply_fn<F>(
        &self,
        buf: &Buffer<T, Self, S>,
        f: impl Fn(Resolve<T>) -> F + Copy,
    ) -> Buffer<T, Self, S>
    where
        F: ToCLSource + Eval<T>,
    {
        let mut out = self.retrieve(buf.len(), buf);

        self.add_op(&mut out, move |out| {
            #[cfg(unified_cl)]
            {
                let cpu_out = unsafe { &mut *(out as *mut Buffer<_, OpenCL<Mods>, _>) };
                self.use_cpu_or_gpu(
                    (file!(), line!(), column!()).into(),
                    &[buf.len()],
                    || crate::devices::cpu_stack_ops::apply_fn_slice(buf, cpu_out, f),
                    || try_cl_apply_fn_mut(self, buf, out, f).unwrap(),
                );
                Ok(())
            }
            #[cfg(not(unified_cl))]
            try_cl_apply_fn_mut(self, buf, out, f)
        }).unwrap();

        out
    }
}

/// A failable OpenCL version of [`apply_fn`](ApplyFunction::apply_fn).
/// It applies a function to a buffer and returns a new buffer.
pub fn try_cl_apply_fn_mut<T, F>(
    device: &CLDevice,
    x: &CLPtr<T>,
    out: &mut CLPtr<T>,
    f: impl Fn(Resolve<T>) -> F,
) -> crate::Result<()>
where
    T: CDatatype + Number,
    F: ToCLSource,
{
    let src = format!(
        "
        __kernel void apply_fn(__global const {datatype}* lhs, __global {datatype}* out, long len) {{
            size_t id = get_global_id(0);
            if (id >= len) {{
                return;
            }}
            out[id] = {operation};
        }}
    ",
        datatype = T::C_DTYPE_STR,
        operation = f("lhs[id]".to_marker()).to_cl_source()
    );

    enqueue_kernel(
        device,
        &src,
        [(x.len() / 32 + 1) * 32, 0, 0],
        Some([32, 0, 0]),
        &[x, out, &x.len()],
    )?;
    Ok(())
}

impl<T, S, Mods: OnDropBuffer + AddOperation<T, Self>> UnaryGrad<T, S> for OpenCL<Mods>
where
    T: CDatatype + Number,
    S: Shape,
{
    #[inline]
    fn add_unary_grad<F>(
        &self,
        lhs: &Buffer<T, Self, S>,
        lhs_grad: &mut Buffer<T, Self, S>,
        out: &Buffer<T, Self, S>,
        lhs_grad_fn: impl Fn(Resolve<T>) -> F + Copy,
    ) where
        F: ToCLSource,
    {
        self.add_op(lhs_grad, move |lhs_grad| {
            try_cl_add_unary_grad(self, lhs, lhs_grad, out, lhs_grad_fn)
        }).unwrap();
    }
}

/// A failable OpenCL version of [`add_unary_grad`](UnaryGrad::add_unary_grad).
/// Writes the unary gradient (with chainrule) to the lhs_grad [`Buffer`].
pub fn try_cl_add_unary_grad<T, S, F, Mods: OnDropBuffer>(
    device: &OpenCL<Mods>,
    lhs: &Buffer<T, OpenCL<Mods>, S>,
    lhs_grad: &mut Buffer<T, OpenCL<Mods>, S>,
    out: &Buffer<T, OpenCL<Mods>, S>,
    lhs_grad_fn: impl Fn(Resolve<T>) -> F,
) -> crate::Result<()>
where
    T: CDatatype + Number,
    F: ToCLSource,
    S: Shape,
{
    let src = format!(
        "
        __kernel void add_unary_grad(__global const {datatype}* lhs, __global {datatype}* lhs_grad, __global const {datatype}* out, long len) {{
            size_t id = get_global_id(0);
            if (id >= len) {{
                return;
            }}
            lhs_grad[id] += out[id] * {operation};
        }}
    ",
        datatype = T::C_DTYPE_STR,
        operation = lhs_grad_fn("lhs[id]".to_marker()).to_cl_source()
    );

    enqueue_kernel(
        device,
        &src,
        [(lhs.len() / 32 + 1) * 32, 0, 0],
        None,
        &[lhs, lhs_grad, out, &out.len()],
    )?;
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        opencl::{chosen_cl_idx, try_cl_add_unary_grad, try_cl_apply_fn_mut},
        ApplyFunction, Base, Buffer, Combiner, OpenCL,
    };

    #[test]
    fn test_cl_apply_fn() -> crate::Result<()> {
        let device = OpenCL::<Base>::new(chosen_cl_idx())?;

        let buf = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));
        let mut out = Buffer::<_, _>::new(&device, buf.len());
        try_cl_apply_fn_mut(&device, &buf, &mut out, |x| x.mul(2))?;
        assert_eq!(out.read(), [2, 4, 6, 8, 10, 12]);

        Ok(())
    }

    #[test]
    fn test_cl_add_unary_grad() -> crate::Result<()> {
        let device = OpenCL::<Base>::new(chosen_cl_idx())?;
        println!("device: {:?}", device.name());
        let lhs = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));
        let mut lhs_grad = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

        let out = Buffer::from((&device, [1, 1, 1, 1, 1, 1]));

        try_cl_add_unary_grad(&device, &lhs, &mut lhs_grad, &out, |x| x.mul(2).add(1))?;

        assert_eq!(lhs_grad.read(), [4, 7, 10, 13, 16, 19]);

        Ok(())
    }

    #[test]
    fn test_cl_apply_fn_autograd() -> crate::Result<()> {
        let device = OpenCL::<crate::Autograd<Base>>::new(chosen_cl_idx())?;
        let lhs = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

        device.apply_fn(&lhs, |x| x.mul(2));

        Ok(())
    }
}
