use custos::{Buffer, CDatatype, Cache, CPU};

#[cfg(feature = "opencl")]
use custos::{opencl::enqueue_kernel, OpenCL};

#[cfg(feature = "cuda")]
use custos::{cuda::launch_kernel1d, CUDA};

/// AddBuf will be implemented for all compute devices.
pub trait AddBuf<T>: Sized {
    /// This operation perfoms element-wise addition.
    fn add(&self, lhs: &Buffer<T, Self>, rhs: &Buffer<T, Self>) -> Buffer<T, Self>;
    // ... you can add more operations if you want to do that.
}

// Host CPU implementation
impl<T> AddBuf<T> for CPU
where
    T: Copy + std::ops::Add<Output = T>,
{
    // you can use the custos::Number trait. This trait is implemented for all number types (usize, i16, f32, ...)
    fn add(&self, lhs: &Buffer<T, CPU>, rhs: &Buffer<T, CPU>) -> Buffer<T, CPU> {
        let len = std::cmp::min(lhs.len, rhs.len);

        // this returns a previously allocated buffer.
        // You can deactivate the caching behaviour by adding the "realloc" feature
        // to the custos feature list in the Cargo.toml.
        let mut out = Cache::get(self, len, [lhs.node.idx, rhs.node.idx]);

        // By default, the Buffer dereferences to a slice.
        // Therefore, standard indexing can be used.
        // You can pass a CPU Buffer to a function that takes a slice as a parameter, too.
        // However, the Buffer must be created via a CPU.
        for i in 0..len {
            out[i] = lhs[i] + rhs[i];
        }
        out
    }
}

#[cfg(feature = "opencl")]
// OpenCL implementation
impl<T> AddBuf<T> for OpenCL
where
    T: CDatatype, // the custos::CDatatype trait is used to
{
    // get the OpenCL C type string for creating generic OpenCL kernels.
    fn add(&self, lhs: &Buffer<T, OpenCL>, rhs: &Buffer<T, OpenCL>) -> Buffer<T, OpenCL> {
        // generic OpenCL kernel
        let src = format!("
            __kernel void add(__global const {datatype}* lhs, __global const {datatype}* rhs, __global {datatype}* out) 
            {{
                size_t id = get_global_id(0);
                out[id] = lhs[id] + rhs[id];
            }}
        ", datatype=T::as_c_type_str());

        let len = std::cmp::min(lhs.len, rhs.len);
        let out = Cache::get::<T, OpenCL>(self, len, [lhs.node.idx, rhs.node.idx]);

        // In the background, the kernel is compiled once. After that, it will be reused for every iteration.
        // The cached kernels are released (or freed) when the underlying CLDevice is dropped.
        // The arguments are specified with a slice of buffers and/or numbers.
        enqueue_kernel(self, &src, [len, 0, 0], None, &[&lhs, &rhs, &out]).unwrap();
        out
    }
}

#[cfg(feature = "cuda")]
// CUDA Implementation
impl<T: CDatatype> AddBuf<T> for CUDA {
    fn add(&self, lhs: &Buffer<T, CUDA>, rhs: &Buffer<T, CUDA>) -> Buffer<T, CUDA> {
        // generic CUDA kernel
        let src = format!(
            r#"extern "C" __global__ void add({datatype}* lhs, {datatype}* rhs, {datatype}* out, int numElements)
                {{
                    int idx = blockDim.x * blockIdx.x + threadIdx.x;
                    if (idx < numElements) {{
                        out[idx] = lhs[idx] + rhs[idx];
                    }}
                    
                }}
        "#,
            datatype = T::as_c_type_str()
        );

        let len = std::cmp::min(lhs.len, rhs.len);
        let out = Cache::get::<T, CUDA>(self, len, (lhs.node.idx, rhs.node.idx));

        // The kernel is compiled once with nvrtc and is cached too.
        // The arguments are specified with a vector of buffers and/or numbers.
        launch_kernel1d(len, self, &src, "add", &[lhs, rhs, &out, &len]).unwrap();
        out
    }
}

pub trait AddOp<'a, T, D> {
    fn add(&self, rhs: &Buffer<'a, T, D>) -> Buffer<'a, T, D>;
}

impl<'a, T: CDatatype, D: AddBuf<T>> AddOp<'a, T, D> for Buffer<'a, T, D> {
    #[inline]
    fn add(&self, rhs: &Buffer<'a, T, D>) -> Buffer<'a, T, D> {
        self.device().add(self, rhs)
    }
}

#[allow(dead_code)]
pub struct OwnStruct<'a, T, D> {
    buf: Buffer<'a, T, D>,
}

impl<'a, T, D> OwnStruct<'a, T, D> {
    #[allow(dead_code)]
    // consider using operator overloading for your own type
    #[inline]
    fn add(&self, rhs: &OwnStruct<T, D>) -> Buffer<T, D>
    where
        T: CDatatype,
        D: AddBuf<T>
    {
        self.buf.device().add(&self.buf, &rhs.buf)
        //get_device!(self.buf.device, AddBuf<T>).add(&self.buf, &rhs.buf)
    }

    // general context
    /*#[inline]
    fn operation(&self, rhs: &OwnStruct<T>, other_arg: &T) -> OwnStruct<T> {
        get_device!(self.buf.device, OperationTrait<T>).operation(self, rhs, other_arg)
    }*/

    // ...
}

fn main() -> custos::Result<()> {
    let cpu = CPU::new();

    let lhs = Buffer::from((&cpu, [1, 3, 5, 3, 2, 6]));
    let rhs = Buffer::from((&cpu, [-1, -12, -6, 3, 2, -1]));

    let out = cpu.add(&lhs, &rhs);
    assert_eq!(out.read(), vec![0, -9, -1, 6, 4, 5]); // to read a CPU Buffer, you can also call .as_slice() on it.

    // without specifying a device
    let out = lhs.add(&rhs);
    assert_eq!(out.read(), vec![0, -9, -1, 6, 4, 5]);

    #[cfg(feature = "opencl")] // deactivate this block if the feature is disabled
    {
        let cl_device = OpenCL::new(0)?;

        let lhs = Buffer::from((&cl_device, [1, 2, 3, 4, 5, 6]));
        let rhs = Buffer::from((&cl_device, [6, 5, 4, 3, 2, 1]));

        let out = cl_device.add(&lhs, &rhs);
        assert_eq!(out.read(), &[7, 7, 7, 7, 7, 7]);
    }

    #[cfg(feature = "cuda")]
    {
        let cuda_device = CUDA::new(0)?;

        let lhs = Buffer::from((&cuda_device, [1., 2., 3., 4., 5., 6.]));
        let rhs = Buffer::from((&cuda_device, [6., 5., 4., 3., 2., 1.]));

        let out = cuda_device.add(&lhs, &rhs);
        assert_eq!(out.read(), &[7., 7., 7., 7., 7., 7.]);
    }

    Ok(())
}