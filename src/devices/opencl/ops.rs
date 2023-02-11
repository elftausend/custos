use core::ops::{Bound, RangeBounds};

use min_cl::api::{
    enqueue_copy_buffer, enqueue_full_copy_buffer, enqueue_read_buffer, enqueue_write_buffer,
    wait_for_event,
};

use crate::{
    ApplyFunction, Buffer, CDatatype, ClearBuf, CopySlice, Device, OpenCL, Read, Resolve, Shape,
    ToMarker, UnaryGrad, WriteBuf,
};

use super::{enqueue_kernel, CLBuffer};

impl<T: CDatatype> ClearBuf<T> for OpenCL {
    #[inline]
    fn clear(&self, buf: &mut Buffer<T, OpenCL>) {
        try_cl_clear(self, buf).unwrap()
    }
}

/// Sets the elements of an OpenCL Buffer to zero.
/// # Example
/// ```
/// use custos::{OpenCL, Buffer, Read, opencl::try_cl_clear};
///
/// fn main() -> Result<(), custos::Error> {
///     let device = OpenCL::new(0)?;
///     let mut lhs = Buffer::<i16, _>::from((&device, [15, 30, 21, 5, 8]));
///     assert_eq!(device.read(&lhs), vec![15, 30, 21, 5, 8]);
///
///     try_cl_clear(&device, &mut lhs)?;
///     assert_eq!(device.read(&lhs), vec![0; 5]);
///     Ok(())
/// }
/// ```
pub fn try_cl_clear<T: CDatatype>(
    device: &OpenCL,
    lhs: &mut Buffer<T, OpenCL>,
) -> crate::Result<()> {
    let src = format!(
        "
        __kernel void clear(__global {datatype}* self) {{
            size_t id = get_global_id(0);
            self[id] = 0;
        }}
    ",
        datatype = T::as_c_type_str()
    );

    let gws = [lhs.len(), 0, 0];
    enqueue_kernel(device, &src, gws, None, &[lhs])?;
    Ok(())
}

impl<T> WriteBuf<T> for OpenCL {
    #[inline]
    fn write(&self, buf: &mut Buffer<T, OpenCL>, data: &[T]) {
        let event =
            unsafe { enqueue_write_buffer(&self.queue(), buf.cl_ptr(), data, true).unwrap() };
        wait_for_event(event).unwrap();
    }

    #[inline]
    fn write_buf(&self, dst: &mut Buffer<T, Self>, src: &Buffer<T, Self>) {
        debug_assert_eq!(dst.len(), src.len());
        enqueue_full_copy_buffer::<T>(&self.queue(), src.cl_ptr(), dst.cl_ptr(), dst.len())
            .unwrap();
    }
}

impl<T, R: RangeBounds<usize>> CopySlice<T, R> for OpenCL {
    fn copy_slice(&self, buf: &Buffer<T, OpenCL>, range: R) -> Buffer<T, Self> {
        let start = match range.start_bound() {
            Bound::Included(start) => *start,
            Bound::Excluded(start) => start + 1,
            Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            Bound::Excluded(end) => *end,
            Bound::Included(end) => end + 1,
            Bound::Unbounded => buf.len(),
        };

        let slice_len = end - start;
        let copied = Buffer::new(self, slice_len);

        enqueue_copy_buffer::<T>(
            &self.queue(),
            buf.ptr.ptr,
            copied.ptr.ptr,
            start,
            0,
            copied.len(),
        )
        .unwrap();

        copied
    }
}

impl<T: Clone + Default> Read<T> for OpenCL {
    #[cfg(not(unified_cl))]
    type Read<'a> = Vec<T> where T: 'a;
    #[cfg(unified_cl)]
    type Read<'a> = &'a [T] where T: 'a;

    #[cfg(not(unified_cl))]
    fn read<'a>(&self, buf: &'a Buffer<T, OpenCL>) -> Self::Read<'a> {
        self.read_to_vec(buf)
    }

    #[cfg(unified_cl)]
    #[inline]
    fn read<'a>(&self, buf: &'a Buffer<T, OpenCL>) -> Self::Read<'a> {
        buf.as_slice()
    }

    #[inline]
    fn read_to_vec(&self, buf: &Buffer<T, OpenCL>) -> Vec<T> {
        try_read_cl_buf_to_vec(self, buf).unwrap()
    }
}

fn try_read_cl_buf_to_vec<T: Clone + Default>(
    device: &OpenCL,
    buf: &Buffer<T, OpenCL>,
) -> crate::Result<Vec<T>> {
    let mut read = vec![T::default(); buf.len()];
    let event = unsafe { enqueue_read_buffer(&device.queue(), buf.cl_ptr(), &mut read, false)? };
    wait_for_event(event).unwrap();
    Ok(read)
}

impl<T, S> ApplyFunction<T, S> for OpenCL
where
    T: CDatatype,
    S: Shape,
{
    #[inline]
    fn apply_fn<F>(
        &self,
        buf: &Buffer<T, Self, S>,
        f: impl Fn(Resolve<T>) -> F,
    ) -> Buffer<T, Self, S>
    where
        F: ToString,
    {
        try_cl_apply_fn(self, buf, f).unwrap()
    }
}

pub fn try_cl_apply_fn<'a, T, S, F: ToString>(
    device: &'a OpenCL,
    x: &CLBuffer<T, S>,
    f: impl Fn(Resolve<T>) -> F,
) -> crate::Result<CLBuffer<'a, T, S>>
where
    T: CDatatype,
    S: Shape,
{
    let src = format!(
        "
        __kernel void apply_fn(__global const {datatype}* lhs, __global {datatype}* out) {{
            size_t id = get_global_id(0);
            out[id] = {operation};
        }}
    ",
        datatype = T::as_c_type_str(),
        operation = f("lhs[id]".to_marker()).to_string()
    );

    let out = device.retrieve::<T, S>(x.len());
    enqueue_kernel(device, &src, [x.len(), 0, 0], None, &[x, &out])?;
    Ok(out)
}

impl<T, S> UnaryGrad<T, S> for OpenCL
where
    T: CDatatype,
    S: Shape,
{
    #[inline]
    fn add_unary_grad<F>(
        &self,
        lhs: &Buffer<T, Self, S>,
        lhs_grad: &mut Buffer<T, Self, S>,
        out: &Buffer<T, Self, S>,
        lhs_grad_fn: impl Fn(Resolve<T>) -> F,
    ) where
        F: ToString,
    {
        try_cl_add_unary_grad(self, lhs, lhs_grad, out, lhs_grad_fn).unwrap();
    }
}

pub fn try_cl_add_unary_grad<T, S, F>(
    device: &OpenCL,
    lhs: &Buffer<T, OpenCL, S>,
    lhs_grad: &mut Buffer<T, OpenCL, S>,
    out: &Buffer<T, OpenCL, S>,
    lhs_grad_fn: impl Fn(Resolve<T>) -> F,
) -> crate::Result<()>
where
    T: CDatatype,
    F: ToString,
    S: Shape,
{
    let src = format!(
        "
        __kernel void add_unary_grad(__global const {datatype}* lhs, __global {datatype}* lhs_grad, __global const {datatype}* out) {{
            size_t id = get_global_id(0);
            lhs_grad[id] += out[id] * {operation};
        }}
    ",
        datatype = T::as_c_type_str(),
        operation = lhs_grad_fn("lhs[id]".to_marker()).to_string()
    );

    enqueue_kernel(device, &src, [lhs.len(), 0, 0], None, &[lhs, lhs_grad, out])?;
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        opencl::{try_cl_add_unary_grad, try_cl_apply_fn},
        Buffer, Combiner, OpenCL,
    };

    #[test]
    fn test_cl_apply_fn() -> crate::Result<()> {
        let device = OpenCL::new(0)?;

        let buf = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

        let out = try_cl_apply_fn(&device, &buf, |x| x.mul(2))?;
        assert_eq!(out.read(), [2, 4, 6, 8, 10, 12]);

        Ok(())
    }

    #[test]
    fn test_cl_add_unary_grad() -> crate::Result<()> {
        let device = OpenCL::new(0)?;

        let lhs = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));
        let mut lhs_grad = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

        let out = Buffer::from((&device, [1, 1, 1, 1, 1, 1]));

        try_cl_add_unary_grad(&device, &lhs, &mut lhs_grad, &out, |x| x.mul(2).add(1))?;

        assert_eq!(lhs_grad.read(), [4, 7, 10, 13, 16, 19]);

        Ok(())
    }
}
