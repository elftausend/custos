use core::ops::{Range, RangeBounds};

use min_cl::{
    CLDevice,
    api::{enqueue_copy_buffer, enqueue_copy_buffers, enqueue_full_copy_buffer},
};

use crate::{
    AddOperation, ApplyFunction, Buffer, CDatatype, ClearBuf, CopySlice, OpenCL, Read, Resolve,
    Retrieve, Retriever, SetOpHint, Shape, ToCLSource, ToMarker, TwoWay, UnaryGrad, Unit,
    UseGpuOrCpu, WrappedData, WriteBuf, ZeroGrad, bounds_to_range, cpu_stack_ops::clear_slice,
    location, op_hint::unary, pass_down_add_operation, prelude::Number,
};

use super::{CLPtr, enqueue_kernel};

/*impl<Mods: OnDropBuffer, T: CDatatype> ClearBuf<T> for OpenCL<Mods> {
    #[inline]
    fn clear(&self, buf: &mut Buffer<T, OpenCL<Mods>>) {
        try_cl_clear(self, buf).unwrap()
    }
}*/

pass_down_add_operation!(OpenCL);

impl<Mods: WrappedData + UseGpuOrCpu, T: CDatatype + Default> ClearBuf<T> for OpenCL<Mods> {
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
                location!(),
                &[buf.len()],
                || clear_slice(cpu_buf),
                || try_cl_clear(self, buf).unwrap(),
            );
        }

        #[cfg(not(unified_cl))]
        try_cl_clear(self, buf).unwrap()
    }
}

impl<Mods: WrappedData, T: CDatatype> ZeroGrad<T> for OpenCL<Mods> {
    #[inline]
    fn zero_grad<S: Shape>(&self, data: &mut Self::Base<T, S>) {
        try_cl_clear(self, data).unwrap()
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
///     assert_eq!(lhs.read(), vec![15, 30, 21, 5, 8]);
///
///     try_cl_clear(&device, &mut lhs)?;
///     assert_eq!(lhs.read(), vec![0; 5]);
///     Ok(())
/// }
/// ```
pub fn try_cl_clear<T: CDatatype>(device: &CLDevice, lhs: &mut CLPtr<T>) -> crate::Result<()> {
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

impl<T: Unit, S: Shape, Mods: WrappedData> WriteBuf<T, S> for OpenCL<Mods> {
    #[inline]
    fn write(&self, buf: &mut Buffer<T, Self, S>, data: &[T]) {
        let event = unsafe { self.device.enqueue_write_buffer(buf.cl_ptr(), data, false) }.unwrap();
        // let event =
        //     unsafe { enqueue_write_buffer(self.queue(), buf.cl_ptr(), data, false, None).unwrap() };
        // self.device.event_wait_list.borrow_mut().push(event);
        // self.device.wait_for_events().unwrap();
        event.wait().unwrap();
    }

    #[inline]
    fn write_buf(&self, dst: &mut Buffer<T, Self, S>, src: &Buffer<T, Self, S>) {
        debug_assert_eq!(dst.len(), src.len());
        let event = unsafe {
            enqueue_full_copy_buffer::<T>(self.queue(), src.cl_ptr(), dst.cl_ptr(), dst.len(), None)
                .unwrap()
        };
        event.wait().unwrap();
    }
}

impl<T: Unit> CopySlice<T> for OpenCL {
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

        let event = unsafe {
            enqueue_copy_buffer::<T>(
                self.queue(),
                source.data.ptr,
                dest.data.ptr,
                source_range.start,
                dest_range.start,
                source_range.end - source_range.start,
                Some(&self.device.event_wait_list.borrow()),
            )
            .unwrap()
        };
        event.wait().unwrap();
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

        unsafe {
            enqueue_copy_buffers::<T, _>(
                self.queue(),
                source.data.ptr,
                dest.data.ptr,
                ranges,
                Some(&self.device.event_wait_list.borrow()),
            )
            .unwrap()
        };
    }
}

impl<Mods, T, S> Read<T, S> for OpenCL<Mods>
where
    Mods: WrappedData + 'static,
    T: Unit + Clone + Default,
    S: Shape,
{
    #[cfg(not(unified_cl))]
    type Read<'a>
        = Vec<T>
    where
        T: 'a;
    #[cfg(unified_cl)]
    type Read<'a>
        = &'a [T]
    where
        T: 'a;

    #[cfg(not(unified_cl))]
    #[inline]
    fn read<'a>(&self, buf: &'a Self::Base<T, S>) -> Self::Read<'a>
    where
        Self: 'a,
    {
        Read::<T, S>::read_to_vec(self, buf)
    }

    #[cfg(unified_cl)]
    #[inline]
    fn read<'a>(&self, buf: &'a Self::Base<T, S>) -> Self::Read<'a>
    where
        Self: 'a,
    {
        use crate::HostPtr;

        unsafe { buf.as_slice() }
    }

    #[inline]
    fn read_to_vec(&self, buf: &Self::Base<T, S>) -> Vec<T> {
        try_read_cl_buf_to_vec(self, buf).unwrap()
    }
}

fn try_read_cl_buf_to_vec<T: Clone + Default>(
    device: &CLDevice,
    buf: &CLPtr<T>,
) -> crate::Result<Vec<T>> {
    let mut read = vec![T::default(); buf.len()];
    let event = unsafe { device.enqueue_read_buffer(buf.ptr, &mut read, false) }?;
    event.wait().unwrap();
    Ok(read)
}

impl<'a, T, S, Mods> ApplyFunction<T, S> for OpenCL<Mods>
where
    T: CDatatype + Number,
    S: Shape,
    Mods: AddOperation + Retrieve<Self, T, S> + UseGpuOrCpu + SetOpHint<T> + 'static,
{
    #[inline]
    fn apply_fn<F>(
        &self,
        buf: &Buffer<T, Self, S>,
        f: impl Fn(Resolve<T>) -> F + Copy + 'static,
    ) -> Buffer<T, Self, S>
    where
        F: TwoWay<T>,
    {
        let mut out = self.retrieve(buf.len(), buf).unwrap();

        self.add_op((&mut out, buf), move |(out, buf)| {
            let dev = buf.device();
            // let out: &mut Buffer<'_, T, OpenCL<Mods>, S> = out.as_mut().unwrap();
            let out = &mut *out;
            #[cfg(unified_cl)]
            {
                let cpu_out = unsafe { &mut *(out as *mut Buffer<_, OpenCL<Mods>, _>) };
                dev.use_cpu_or_gpu(
                    (file!(), line!(), column!()).into(),
                    &[buf.len()],
                    || crate::devices::cpu_stack_ops::apply_fn_slice(buf, cpu_out, f),
                    || try_cl_apply_fn_mut(dev, buf, out, f).unwrap(),
                );
                Ok(())
            }
            #[cfg(not(unified_cl))]
            {
                try_cl_apply_fn_mut(dev, buf, out, f)?;
                Ok(())
            }
        })
        .unwrap();
        self.set_op_hint(unary(f));
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

impl<T, S, Mods: WrappedData + AddOperation + 'static> UnaryGrad<T, S> for OpenCL<Mods>
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
        lhs_grad_fn: impl Fn(Resolve<T>) -> F + Copy + 'static,
    ) where
        F: ToCLSource,
    {
        self.add_op((lhs, lhs_grad, out), move |(lhs, lhs_grad, out)| {
            try_cl_add_unary_grad(lhs.device(), lhs, lhs_grad, out, lhs_grad_fn)
        })
        .unwrap();
    }
}

/// A failable OpenCL version of [`add_unary_grad`](UnaryGrad::add_unary_grad).
/// Writes the unary gradient (with chainrule) to the lhs_grad [`Buffer`].
pub fn try_cl_add_unary_grad<T, S, F, Mods: WrappedData>(
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
        ApplyFunction, Base, Buffer, Combiner, OpenCL,
        opencl::{chosen_cl_idx, try_cl_add_unary_grad, try_cl_apply_fn_mut},
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

    #[cfg(feature = "autograd")]
    #[test]
    fn test_cl_apply_fn_autograd() -> crate::Result<()> {
        let device = OpenCL::<crate::Autograd<Base>>::new(chosen_cl_idx())?;
        let lhs = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

        device.apply_fn(&lhs, |x| x.mul(2));

        Ok(())
    }

    #[cfg(feature = "lazy")]
    #[test]
    fn test_cl_lazy_unary_grad_exec() {
        use crate::{Lazy, Run, UnaryGrad};

        let device = OpenCL::<Lazy<Base>>::new(0).unwrap();
        let lhs = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));
        let mut lhs_grad = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

        let out = Buffer::from((&device, [1, 1, 1, 1, 1, 1]));

        device.add_unary_grad(&lhs, &mut lhs_grad, &out, |x| x.mul(2).add(1));
        device.run().unwrap();

        assert_eq!(lhs_grad.read(), [4, 7, 10, 13, 16, 19]);
    }
}
