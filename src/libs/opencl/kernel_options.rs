use crate::{matrix::Matrix, number::Number, Error, GenericOCL, Buffer, Node};

use super::{api::{enqueue_nd_range_kernel, set_kernel_arg, OCLErrorKind}, CL_CACHE, CLCache, cl_device::InternCLDevice};

pub trait KernelArg<'a, T> {
    fn buf(&'a self) -> Option<&'a Buffer<T>> {
        None
    }
    fn number(&self) -> Option<T> {
        None
    }
}

impl<'a, T: Copy> KernelArg<'a, T> for Matrix<T> {
    fn buf(&'a self) -> Option<&'a Buffer<T>> {
        Some(self.as_buf())
    }
}

impl<'a, T: Copy> KernelArg<'a, T> for &'a Matrix<T> {
    fn buf(&self) -> Option<&'a Buffer<T>> {
        Some(self.as_buf())
    }
}

impl<'a, T: Copy> KernelArg<'a, T> for Buffer<T> {
    fn buf(&'a self) -> Option<&'a Buffer<T>> {
        Some(self)
    }
}

impl<'a, T: Copy> KernelArg<'a, T> for &'a Buffer<T> {
    fn buf(&self) -> Option<&'a Buffer<T>> {
        Some(self)
    }
}

impl<'a, T: Number> KernelArg<'a, T> for T {
    fn number(&self) -> Option<T> {
        Some(*self)
    }
}
/// Provides an API to run and cache OpenCL kernels.
/// 
/// # Errors
/// OpenCL related errors
/// 
/// # Note
/// 
/// if with_output(...) is not provided, the output buffer is the lhs buffer.
/// 
/// # Example
/// ```
/// use custos::{opencl::KernelOptions, CLDevice, Error, GenericOCL, VecRead, Buffer};
/// 
/// fn main() -> Result<(), Error> {
///     let device = CLDevice::get(0)?;
/// 
///     let lhs = Buffer::<i32>::from((&device, [1, 5, 3, 2, 7, 8]));
///     let rhs = Buffer::<i32>::from((&device, [-2, -6, -4, -3, -8, -9]));
/// 
///     let src = format!("
///         __kernel void add(__global {datatype}* self, __global const {datatype}* rhs, __global {datatype}* out) {{
///             size_t id = get_global_id(0);
///             out[id] = self[id]+rhs[id];
///         }}
///     ", datatype=i32::as_ocl_type_str());
/// 
///     let gws = [lhs.len, 0, 0];
///     let out = KernelOptions::<i32>::new(&device, &lhs, gws, &src)?
///         .with_rhs(&rhs)
///         .with_output(lhs.len)
///         .run()?;
/// 
///     assert_eq!(device.read(&out), vec![-1, -1, -1, -1, -1, -1]);
///     Ok(())
/// }
/// ```
pub struct KernelOptions<'a, T> {
    src: &'a str,
    lhs: &'a Buffer<T>,
    rhs: Option<&'a Buffer<T>>,
    output: Option<Buffer<T>>,
    tensor_args: Vec<(&'a Buffer<T>, usize)>,
    number_args: Vec<(T, usize)>,
    gws: [usize; 3],
    lws: Option<[usize; 3]>,
    offset: Option<[usize; 3]>,
    wd: usize,
    device: InternCLDevice,
}

impl<'a, T: GenericOCL> KernelOptions<'a, T> {
    pub fn new(device: &InternCLDevice, lhs: &'a Buffer<T>, gws: [usize; 3], src: &'a str) -> crate::Result<KernelOptions<'a, T>> {

        let wd;
        if gws[0] == 0 {
            return Err(OCLErrorKind::InvalidGlobalWorkSize.into());
        } else if gws[1] == 0 {
            wd=1;
        } else if gws[2] == 0 {
            wd=2;
        } else {
            wd=3;
        }
        let tensor_args = vec![(lhs, 0)];
        Ok(KernelOptions {
            src,
            lhs,
            rhs: None,
            output: None,
            tensor_args,
            number_args: Vec::new(),
            gws,
            lws: None,
            offset: None,
            wd,
            device: device.clone(),
        })
    }
    pub fn with_lws(&mut self, lws: [usize; 3]) -> &mut KernelOptions<'a, T> {
        self.lws = Some(lws);
        self
    }

    pub fn with_offset(&mut self, offset: [usize; 3]) -> &mut Self {
        self.offset = Some(offset);
        self
    }

    /// Sets buffer to index 1 for the kernel argument list
    pub fn with_rhs(&mut self, rhs: &'a Buffer<T>) -> &mut KernelOptions<'a, T> {
        self.tensor_args.push((rhs, 1));
        self.rhs = Some(rhs);
        self
    }
    /// Adds value (Matrix<T> or T) to the kernel argument list
    pub fn add_arg<A: KernelArg<'a, T>>(&'a mut self, arg: &'a A) -> &mut KernelOptions<'a, T> {
        let idx = self.number_args.len()+self.tensor_args.len();
        
        match arg.number() {
            Some(number) => self.number_args.push((number, idx)),
            None => self.tensor_args.push((arg.buf().unwrap(), idx)),
        }
        self
    }

    /// Adds output
    pub fn with_output(&mut self, out_len: usize) -> &mut KernelOptions<'a, T> {
        self.output = Some(CLCache::get(self.device.clone(), Node::new(out_len)));
        self
    }

    /// Runs the kernel
    pub fn run(&'a mut self) -> Result<Buffer<T>, Error> {
        let kernel = CL_CACHE.with(|cache| cache.borrow_mut().arg_kernel_cache(self.device.clone(), &self.tensor_args, &self.number_args, self.output.as_ref(), self.src.to_string()))?;
        
        for arg in &self.number_args {
            set_kernel_arg(&kernel, arg.1, &arg.0)?
        }
    
        enqueue_nd_range_kernel(&self.device.queue(), &kernel, self.wd, &self.gws, self.lws.as_ref(), self.offset)?;
    
        match &self.output {
            Some(out) => Ok(out.clone()),
            None => Ok(self.lhs.clone()),
        }
    }
}
