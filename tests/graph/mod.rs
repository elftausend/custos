use std::ops::Deref;

use custos::{number::Number, Buffer, CDatatype, Device, CPU, Retriever, Retrieve};

#[cfg(feature = "opencl")]
use custos::{opencl::enqueue_kernel, OpenCL};

#[cfg(not(feature = "realloc"))]
mod graph;

#[cfg(not(feature = "realloc"))]
#[cfg(unified_cl)]
mod to_unified;

#[cfg(feature = "cuda")]
use custos::CUDA;

pub trait AddBuf<T, D: Device>: Device {
    fn add(&self, lhs: &Buffer<T, D>, rhs: &Buffer<T, D>) -> Buffer<T, Self>;
    fn relu(&self, lhs: &Buffer<T, D>) -> Buffer<T, Self>;
}

impl<T, D, Mods> AddBuf<T, D> for CPU<Mods>
where
    Mods: Retrieve<Self, T>,
    D: Device,
    D::Data<T, ()>: Deref<Target = [T]>,
    T: Number,
{
    fn add(&self, lhs: &Buffer<T, D>, rhs: &Buffer<T, D>) -> Buffer<T, Self> {
        let len = std::cmp::min(lhs.len(), rhs.len());

        let mut out = self.retrieve(lhs.len(), (lhs, rhs));

        for i in 0..len {
            out[i] = lhs[i] + rhs[i];
        }
        out
    }

    fn relu(&self, lhs: &Buffer<T, D>) -> Buffer<T, Self> {
        let mut out = self.retrieve(lhs.len(), lhs);

        for i in 0..lhs.len() {
            if lhs[i] > T::zero() {
                out[i] = lhs[i];
            }
        }
        out
    }
}

#[cfg(feature = "opencl")]
impl<T: CDatatype, Mods: Retrieve<Self, T>> AddBuf<T, Self> for OpenCL<Mods> {
    fn add(&self, lhs: &Buffer<T, Self>, rhs: &Buffer<T, Self>) -> Buffer<T, Self> {
        let src = format!("
        __kernel void add(__global {datatype}* self, __global const {datatype}* rhs, __global {datatype}* out) {{
            size_t id = get_global_id(0);
            out[id] = self[id] + rhs[id];
        }}
    ", datatype=T::C_DTYPE_STR);

        let gws = [lhs.len(), 0, 0];
        let out = self.retrieve(lhs.len(), (lhs, rhs));
        enqueue_kernel(self, &src, gws, None, &[lhs, rhs, &out.data]).unwrap();
        out
    }

    fn relu(&self, lhs: &Buffer<T, Self>) -> Buffer<T, Self> {
        let src = format!(
            "
            __kernel void str_op(__global const {datatype}* lhs, __global {datatype}* out) {{
                size_t id = get_global_id(0);
                out[id] = max(lhs[id], 0);
            }}
        ",
            datatype = T::C_DTYPE_STR
        );

        let out = self.retrieve(lhs.len(), lhs);
        enqueue_kernel(self, &src, [lhs.len(), 0, 0], None, &[lhs, &out.data]).unwrap();
        out
    }
}

#[cfg(feature = "cuda")]
impl<T: CDatatype, Mods: Retrieve<Self, T>> AddBuf<T, Self> for CUDA<Mods> {
    fn add(&self, lhs: &Buffer<T, Self>, rhs: &Buffer<T, Self>) -> Buffer<T, Self> {
        let src = format!(
            r#"extern "C" __global__ void add({datatype}* lhs, {datatype}* rhs, {datatype}* out, int numElements)
                {{
                    int idx = blockDim.x * blockIdx.x + threadIdx.x;
                    if (idx < numElements) {{
                        out[idx] = lhs[idx] + rhs[idx];
                    }}
                  
                }}
        "#,
            datatype = T::C_DTYPE_STR
        );

        let out = self.retrieve(lhs.len(), (lhs, rhs));
        self.launch_kernel1d(lhs.len, &src, "add", &[lhs, rhs, &out.data, &lhs.len]).unwrap();
        out
    }

    fn relu(&self, lhs: &Buffer<T, Self>) -> Buffer<T, Self> {
        let src = format!(
            r#"extern "C" __global__ void relu({datatype}* lhs, {datatype}* out, int numElements)
                {{
                    int idx = blockDim.x * blockIdx.x + threadIdx.x;
                    if (idx < numElements) {{
                        out[idx] = max(lhs[idx], 0);
                    }}
                  
                }}
        "#,
            datatype = T::C_DTYPE_STR
        );

        let out = self.retrieve(lhs.len(), lhs);
        self.launch_kernel1d(lhs.len, &src, "add", &[lhs, &out.data, &lhs.len]).unwrap();
        out
    }
}

pub trait AddOp<'a, T, D: Device> {
    fn add(&self, rhs: &Buffer<'a, T, D>) -> Buffer<'a, T, D>;
    fn relu(&self) -> Buffer<'a, T, D>;
}

impl<'a, T: CDatatype, D: AddBuf<T, D>> AddOp<'a, T, D> for Buffer<'a, T, D> {
    #[inline]
    fn add(&self, rhs: &Buffer<'a, T, D>) -> Buffer<'a, T, D> {
        self.device().add(self, rhs)
    }

    fn relu(&self) -> Buffer<'a, T, D> {
        self.device().relu(self)
    }
}
