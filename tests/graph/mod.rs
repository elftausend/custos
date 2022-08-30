use custos::{get_device, number::Number, Buffer, Cache, CPU, CDatatype};

#[cfg(feature="opencl")]
use custos::{CLDevice, opencl::enqueue_kernel};

mod graph;

#[cfg(feature="opencl")]
mod to_unified;

#[cfg(feature="cuda")]
use custos::{CudaDevice, cuda::launch_kernel1d};

pub trait AddBuf<T> {
    fn add(&self, lhs: &Buffer<T>, rhs: &Buffer<T>) -> Buffer<T>;
    fn relu(&self, lhs: &Buffer<T>) -> Buffer<T>;
}

impl<T> AddBuf<T> for CPU
where
    T: Number,
{
    fn add(&self, lhs: &Buffer<T>, rhs: &Buffer<T>) -> Buffer<T> {
        let len = std::cmp::min(lhs.len, rhs.len);

        let mut out = Cache::get(self, len, (lhs.node.idx, rhs.node.idx));

        for i in 0..len {
            out[i] = lhs[i] + rhs[i];
        }
        out
    }

    fn relu(&self, lhs: &Buffer<T>) -> Buffer<T> {
        let mut out = Cache::get(self, lhs.len, (lhs.node.idx, lhs.node.idx));

        for i in 0..lhs.len {
            if lhs[i] > T::zero() {
                out[i] = lhs[i];
            }
        }
        out
    }
}

#[cfg(feature="opencl")]
impl<T: CDatatype> AddBuf<T> for CLDevice {
    fn add(&self, lhs: &Buffer<T>, rhs: &Buffer<T>) -> Buffer<T> {
        let src = format!("
        __kernel void add(__global {datatype}* self, __global const {datatype}* rhs, __global {datatype}* out) {{
            size_t id = get_global_id(0);
            out[id] = self[id] + rhs[id];
        }}
    ", datatype=T::as_c_type_str());

        let gws = [lhs.len, 0, 0];
        let out = Cache::get::<T, _>(self, lhs.len, (lhs.node.idx, rhs.node.idx));
        enqueue_kernel(self, &src, gws, None, &[lhs, rhs, &out]).unwrap();
        out
    }


    fn relu(&self, lhs: &Buffer<T>) -> Buffer<T> {
        let src = format!(
            "
            __kernel void str_op(__global const {datatype}* lhs, __global {datatype}* out) {{
                size_t id = get_global_id(0);
                out[id] = max(lhs[id], 0);
            }}
        ",
            datatype = T::as_c_type_str()
        );
    
        let out = Cache::get::<T, _>(self, lhs.len(), lhs.node.idx);
        enqueue_kernel(self, &src, [lhs.len, 0, 0], None, &[lhs, &out]).unwrap();
        out
    }
}

#[cfg(feature="cuda")]
impl<T: CDatatype> AddBuf<T> for CudaDevice {
    fn add(&self, lhs: &Buffer<T>, rhs: &Buffer<T>) -> Buffer<T> {
        let src = format!(
            r#"extern "C" __global__ void add({datatype}* lhs, {datatype}* rhs, {datatype}* out, int numElements)
                {{
                    int idx = blockDim.x * blockIdx.x + threadIdx.x;
                    if (idx < numElements) {{
                        out[idx] = lhs[idx] + rhs[idx];
                    }}
                  
                }}
        "#, datatype = T::as_c_type_str());

        let out = Cache::get::<T, _>(self, lhs.len, (lhs.node.idx, rhs.node.idx));
        launch_kernel1d(lhs.len, self, &src, "add", &[
            lhs, rhs, &out, &lhs.len
        ]).unwrap();
        out
    }


    fn relu(&self, lhs: &Buffer<T>) -> Buffer<T> {
        let src = format!(
            r#"extern "C" __global__ void relu({datatype}* lhs, {datatype}* out, int numElements)
                {{
                    int idx = blockDim.x * blockIdx.x + threadIdx.x;
                    if (idx < numElements) {{
                        out[idx] = max(lhs[idx], 0);
                    }}
                  
                }}
        "#, datatype = T::as_c_type_str());
    
        let out = Cache::get::<T, _>(self, lhs.len(), lhs.node.idx);
        launch_kernel1d(lhs.len, self, &src, "relu", &[
            lhs, &out, &lhs.len
        ]).unwrap();
        out
    }
}

pub trait AddOp<'a, T> {
    fn add(&self, rhs: &Buffer<'a, T>) -> Buffer<'a, T>;
    fn relu(&self) -> Buffer<'a, T>;
}

impl<'a, T: CDatatype> AddOp<'a, T> for Buffer<'a, T> {
    #[inline]
    fn add(&self, rhs: &Buffer<'a, T>) -> Buffer<'a, T> {
        get_device!(self.device, AddBuf<T>).add(self, rhs)
    }

    fn relu(&self) -> Buffer<'a, T> {
        get_device!(self.device, AddBuf<T>).relu(self)
    }
}