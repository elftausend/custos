use core::ops::{Range, RangeBounds};

use crate::{
    bounds_to_range,
    cuda::api::{cu_read_async, CUstreamCaptureStatus},
    op_hint::unary,
    pass_down_add_operation, pass_down_exec_now, AddOperation, ApplyFunction, AsNoId, BufAsNoId,
    Buffer, CDatatype, ClearBuf, CopySlice, OnDropBuffer, Read, Resolve, Retrieve, Retriever,
    SetOpHint, Shape, ToCLSource, ToMarker, UnaryGrad, WriteBuf, ZeroGrad, CUDA,
};

use super::{
    api::{cuMemcpy, cu_write_async},
    cu_clear, CUDAPtr, CudaDevice,
};

pass_down_add_operation!(CUDA);
pass_down_exec_now!(CUDA);

impl<Mods: OnDropBuffer, T: Default + Clone, S: Shape> Read<T, S> for CUDA<Mods> {
    type Read<'a> = Vec<T>
    where
        T: 'a,
        CUDA<Mods>: 'a;

    #[inline]
    fn read<'a>(&self, buf: &'a Self::Base<T, S>) -> Vec<T>
    where
        CUDA<Mods>: 'a,
    {
        Read::<T, S>::read_to_vec(self, buf)
    }

    fn read_to_vec(&self, buf: &Self::Base<T, S>) -> Vec<T>
    where
        T: Default + Clone,
    {
        assert!(buf.ptr != 0, "called Read::read(..) on a non CUDA buffer");
        // TODO: sync here or somewhere else?
        if self.stream().capture_status().unwrap()
            == CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            self.stream().sync().unwrap();
        }

        let mut read = vec![T::default(); buf.len];
        cu_read_async(&mut read, buf.ptr, &self.mem_transfer_stream).unwrap();
        self.mem_transfer_stream.sync().unwrap();
        read
    }
}

impl<Mods: OnDropBuffer, T: CDatatype> ClearBuf<T> for CUDA<Mods> {
    #[inline]
    fn clear(&self, buf: &mut Buffer<T, Self>) {
        cu_clear(self, buf).unwrap()
    }
}

impl<Mods: OnDropBuffer, T: CDatatype> ZeroGrad<T> for CUDA<Mods> {
    #[inline]
    fn zero_grad<S: Shape>(&self, data: &mut Self::Base<T, S>) {
        cu_clear(self, data).unwrap()
    }
}

impl<Mods: OnDropBuffer, T> CopySlice<T> for CUDA<Mods> {
    fn copy_slice_to<SR: RangeBounds<usize>, DR: RangeBounds<usize>>(
        &self,
        source: &Buffer<T, Self>,
        source_range: SR,
        dest: &mut Buffer<T, Self>,
        dest_range: DR,
    ) {
        let source_range = bounds_to_range(source_range, source.len());
        let dest_range = bounds_to_range(dest_range, dest.len());

        let len = source_range.end - source_range.start;
        assert_eq!(len, dest_range.end - dest_range.start);
        let size = std::mem::size_of::<T>();

        unsafe {
            cuMemcpy(
                dest.base().ptr + (dest_range.start * size) as u64,
                source.base().ptr + (source_range.start * size) as u64,
                len * size,
            );
        }
    }

    fn copy_slice_all<I: IntoIterator<Item = (Range<usize>, Range<usize>)>>(
        &self,
        source: &Buffer<T, Self>,
        dest: &mut Buffer<T, Self>,
        ranges: I,
    ) {
        for (source_range, dest_range) in ranges {
            self.copy_slice_to(source, source_range, dest, dest_range);
        }
    }
}

impl<Mods: OnDropBuffer, T> WriteBuf<T> for CUDA<Mods> {
    #[inline]
    fn write(&self, buf: &mut Buffer<T, Self>, data: &[T]) {
        cu_write_async(buf.cu_ptr(), data, &self.mem_transfer_stream).unwrap();
    }

    #[inline]
    fn write_buf(&self, dst: &mut Buffer<T, Self, ()>, src: &Buffer<T, Self, ()>) {
        unsafe {
            cuMemcpy(
                dst.base().ptr,
                src.base().ptr,
                src.len() * std::mem::size_of::<T>(),
            );
        }
    }
}

impl<Mods, T, S> ApplyFunction<T, S> for CUDA<Mods>
where
    T: CDatatype + Default,
    Mods: AddOperation + Retrieve<Self, T, S> + SetOpHint<T> + 'static,
    S: Shape,
{
    #[inline]
    fn apply_fn<F>(
        &self,
        buf: &Buffer<T, Self, S>,
        f: impl Fn(Resolve<T>) -> F + Copy + 'static,
    ) -> Buffer<T, Self, S>
    where
        F: crate::TwoWay<T>,
    {
        let mut out = self.retrieve(buf.len(), buf);
        self.add_op((&mut out, buf, f.no_id()), |(out, buf, f)| {
            try_cu_apply_fn_mut(buf.device(), buf, out, &**f)
        })
        .unwrap();
        self.set_op_hint(unary(f));
        out
    }
}

pub fn try_cu_apply_fn_mut<T, F>(
    device: &CudaDevice,
    x: &CUDAPtr<T>,
    out: &mut CUDAPtr<T>,
    f: impl Fn(Resolve<T>) -> F,
) -> crate::Result<()>
where
    F: ToCLSource,
    T: CDatatype + Default,
{
    let src = format!(
        r#"extern "C" __global__ void applyFn({datatype}* x, {datatype}* out, int numElements)
            {{
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx >= numElements) {{
                    return;
                }}
                out[idx] = {op};
            }}
    "#,
        datatype = T::C_DTYPE_STR,
        op = f("x[idx]".to_marker()).to_cl_source()
    );

    device.launch_kernel(
        &src,
        "applyFn",
        [(x.len as u32 / 32 + 1) * 32, 1, 1],
        [32, 1, 1],
        0,
        &[x, out, &x.len],
    )?;
    Ok(())
}

impl<T, S, Mods> UnaryGrad<T, S> for CUDA<Mods>
where
    T: CDatatype + Default,
    S: Shape,
    Mods: OnDropBuffer + AddOperation + 'static,
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
        self.add_op(
            (lhs, lhs_grad.buf_no_id(), out, lhs_grad_fn.no_id()),
            move |(lhs, lhs_grad, out, lhs_grad_fn)| {
                try_cu_add_unary_grad(lhs.device(), lhs, lhs_grad, out, **lhs_grad_fn)
            },
        )
        .unwrap();
    }
}
pub fn try_cu_add_unary_grad<T, F>(
    device: &CudaDevice,
    lhs: &CUDAPtr<T>,
    lhs_grad: &mut CUDAPtr<T>,
    out: &CUDAPtr<T>,
    lhs_grad_fn: impl Fn(Resolve<T>) -> F,
) -> crate::Result<()>
where
    T: CDatatype + Default,
    F: ToCLSource,
{
    let src = format!(
        r#"
        extern "C" __global__ void addUnaryGrad({dtype}* lhs, {dtype}* lhsGrad, {dtype}* out, int numElements)
            {{
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx >= numElements) {{
                    return;
                }}
                lhsGrad[idx] += out[idx] * {op};
            }}
    "#,
        dtype = T::C_DTYPE_STR,
        op = lhs_grad_fn("lhs[idx]".to_marker()).to_cl_source()
    );
    device.launch_kernel1d(
        lhs.len,
        &src,
        "addUnaryGrad",
        &[lhs, lhs_grad, out, &lhs.len],
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{
        cuda::ops::{try_cu_add_unary_grad, try_cu_apply_fn_mut},
        Base, Buffer, Combiner, CUDA,
    };

    #[test]
    fn test_cu_apply_fn() {
        let device = CUDA::<Base>::new(0).unwrap();
        let x = Buffer::from((&device, [1f32, 2., 3., 4., 5., 6.]));
        let mut out = x.empty_like();
        try_cu_apply_fn_mut(&device, &x.data, &mut out.data, |x| x.add("1.0")).unwrap();
        assert_eq!(out.read(), [2f32, 3., 4., 5., 6., 7.,])
    }

    #[test]
    fn test_cu_add_unary_grad() -> crate::Result<()> {
        let device = CUDA::<Base>::new(0)?;
        let lhs = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));
        let mut lhs_grad = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

        let out = Buffer::from((&device, [1, 1, 1, 1, 1, 1]));

        try_cu_add_unary_grad(&device, &lhs.data, &mut lhs_grad.data, &out.data, |x| {
            x.mul(2).add(1)
        })?;

        assert_eq!(lhs_grad.read(), [4, 7, 10, 13, 16, 19]);

        Ok(())
    }

    #[cfg(feature = "lazy")]
    #[test]
    fn test_cu_add_unary_grad_lazy_graph() {
        use crate::{Lazy, Run, UnaryGrad};

        let device = CUDA::<Lazy<Base>>::new(0).unwrap();

        let lhs = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));
        let mut lhs_grad = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

        let out = Buffer::from((&device, [1, 1, 1, 1, 1, 1]));
        device.add_unary_grad(&lhs, &mut lhs_grad, &out, |lhs| lhs.add(2));

        assert_eq!(lhs_grad.read(), vec![1, 2, 3, 4, 5, 6]);

        unsafe { device.run().unwrap() }

        assert_eq!(lhs_grad.read(), vec![4, 6, 8, 10, 12, 14]);
    }
}
