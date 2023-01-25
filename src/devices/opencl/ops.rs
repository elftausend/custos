use min_cl::api::{enqueue_read_buffer, enqueue_write_buffer, wait_for_event};

use crate::{
    ApplyFunction, Buffer, CDatatype, ClearBuf, Device, OpenCL, Read, Resolve, Shape, ToMarker,
    WriteBuf,
};

use super::{enqueue_kernel, CLBuffer};

impl<T: CDatatype> ClearBuf<T, OpenCL> for OpenCL {
    #[inline]
    fn clear(&self, buf: &mut Buffer<T, OpenCL>) {
        cl_clear(self, buf).unwrap()
    }
}

/// Sets the elements of an OpenCL Buffer to zero.
/// # Example
/// ```
/// use custos::{OpenCL, Buffer, Read, opencl::cl_clear};
///
/// fn main() -> Result<(), custos::Error> {
///     let device = OpenCL::new(0)?;
///     let mut lhs = Buffer::<i16, _>::from((&device, [15, 30, 21, 5, 8]));
///     assert_eq!(device.read(&lhs), vec![15, 30, 21, 5, 8]);
///
///     cl_clear(&device, &mut lhs);
///     assert_eq!(device.read(&lhs), vec![0; 5]);
///     Ok(())
/// }
/// ```
pub fn cl_clear<T: CDatatype>(device: &OpenCL, lhs: &mut Buffer<T, OpenCL>) -> crate::Result<()> {
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

impl<T> WriteBuf<T, OpenCL> for OpenCL {
    fn write(&self, buf: &mut Buffer<T, OpenCL>, data: &[T]) {
        let event =
            unsafe { enqueue_write_buffer(&self.queue(), buf.cl_ptr(), data, true).unwrap() };
        wait_for_event(event).unwrap();
    }
}

impl<T: Clone + Default> Read<T, OpenCL> for OpenCL {
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
        read_cl_buf_to_vec(self, buf).unwrap()
    }
}

fn read_cl_buf_to_vec<T: Clone + Default>(
    device: &OpenCL,
    buf: &Buffer<T, OpenCL>,
) -> crate::Result<Vec<T>> {
    let mut read = vec![T::default(); buf.len()];
    let event = unsafe { enqueue_read_buffer(&device.queue(), buf.cl_ptr(), &mut read, false)? };
    wait_for_event(event).unwrap();
    Ok(read)
}

impl<T, S> ApplyFunction<T, S, OpenCL> for OpenCL
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
        cl_apply_fn(self, buf, f).unwrap()
    }
}

pub fn cl_apply_fn<'a, T, S, F: ToString>(
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
