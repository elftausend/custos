use std::ffi::c_void;

use super::{
    api::{enqueue_nd_range_kernel, set_kernel_arg, set_kernel_arg_ptr, OCLErrorKind},
    CLCache, CL_CACHE,
};
use crate::{number::Number, Buffer, CDatatype, CLDevice, CacheBuffer};

pub trait KernelArg<'a, T> {
    fn some_buf(&'a self) -> Option<&'a Buffer<T>> {
        None
    }
    fn number(&self) -> Option<T> {
        None
    }
    fn as_number(&'a self) -> Option<&'a T> {
        None
    }
}

impl<'a, T: Copy> KernelArg<'a, T> for Buffer<T> {
    fn some_buf(&'a self) -> Option<&'a Buffer<T>> {
        Some(self)
    }
}

impl<'a, T: Copy> KernelArg<'a, T> for &'a Buffer<T> {
    fn some_buf(&self) -> Option<&'a Buffer<T>> {
        Some(self)
    }
}

impl<'a, T: Number> KernelArg<'a, T> for T {
    fn number(&self) -> Option<T> {
        Some(*self)
    }

    fn as_number(&'a self) -> Option<&'a T> {
        Some(self)
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
/// use custos::{opencl::KernelOptions, CLDevice, Error, CDatatype, VecRead, Buffer};
///
/// fn main() -> Result<(), Error> {
///     let device = CLDevice::new(0)?;
///
///     let lhs = Buffer::<i32>::from((&device, [1, 5, 3, 2, 7, 8]));
///     let rhs = Buffer::<i32>::from((&device, [-2, -6, -4, -3, -8, -9]));
///
///     let src = format!("
///         __kernel void add(__global {datatype}* self, __global const {datatype}* rhs, __global {datatype}* out) {{
///             size_t id = get_global_id(0);
///             out[id] = self[id]+rhs[id];
///         }}
///     ", datatype=i32::as_c_type_str());
///
///     let gws = [lhs.len, 0, 0];
///     let out = KernelOptions::<i32>::new(&device, &lhs, gws, &src)?
///         .with_rhs(&rhs)
///         .with_output(lhs.len)
///         .run()?.unwrap();
///
///     assert_eq!(device.read(&out), vec![-1, -1, -1, -1, -1, -1]);
///     Ok(())
/// }
/// ```
pub struct KernelOptions<'a, T> {
    src: &'a str,
    output: Option<CacheBuffer<T>>,
    buf_args: Vec<(&'a Buffer<T>, usize)>,
    number_args: Vec<(T, usize)>,
    gws: [usize; 3],
    lws: Option<[usize; 3]>,
    offset: Option<[usize; 3]>,
    wd: usize,
    device: CLDevice,
}

impl<'a, T: CDatatype> KernelOptions<'a, T> {
    pub fn new(
        device: &CLDevice,
        lhs: &'a Buffer<T>,
        gws: [usize; 3],
        src: &'a str,
    ) -> crate::Result<KernelOptions<'a, T>> {
        let wd;
        if gws[0] == 0 {
            return Err(OCLErrorKind::InvalidGlobalWorkSize.into());
        } else if gws[1] == 0 {
            wd = 1;
        } else if gws[2] == 0 {
            wd = 2;
        } else {
            wd = 3;
        }

        Ok(KernelOptions {
            src,
            output: None,
            buf_args: vec![(lhs, 0)],
            number_args: Vec::new(),
            gws,
            lws: None,
            offset: None,
            wd,
            device: device.clone(),
        })
    }

    pub fn with_lws(mut self, lws: [usize; 3]) -> KernelOptions<'a, T> {
        self.lws = Some(lws);
        self
    }

    pub fn with_offset(mut self, offset: [usize; 3]) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Sets buffer to index 1 for the kernel argument list
    pub fn with_rhs(mut self, rhs: &'a Buffer<T>) -> KernelOptions<'a, T> {
        self.buf_args.push((rhs, 1));
        self
    }
    /// Adds value (Matrix<T> or T) to the kernel argument list
    pub fn add_arg<A: KernelArg<'a, T>>(mut self, arg: &'a A) -> KernelOptions<'a, T> {
        let idx = self.number_args.len() + self.buf_args.len();

        match arg.number() {
            Some(number) => self.number_args.push((number, idx)),
            None => self.buf_args.push((arg.some_buf().unwrap(), idx)),
        }
        self
    }

    /// Adds output
    pub fn with_output(mut self, out_len: usize) -> KernelOptions<'a, T> {
        self.output = Some(CLCache::get(&self.device, out_len));
        self
    }

    /// Runs the kernel
    pub fn run(self) -> crate::Result<Option<Buffer<T>>> {
        let kernel = CL_CACHE.with(|cache| {
            cache.borrow_mut().arg_kernel_cache(
                &self.device.clone(),
                &self.buf_args,
                &self.number_args,
                self.output.as_ref(),
                self.src.to_string(),
            )
        })?;

        for arg in &self.number_args {
            set_kernel_arg(&kernel, arg.1, &arg.0)?
        }

        enqueue_nd_range_kernel(
            &self.device.queue(),
            &kernel,
            self.wd,
            &self.gws,
            self.lws.as_ref(),
            self.offset,
        )?;

        if let Some(output) = self.output {
            return Ok(Some(output.to_buf()));
        }
        Ok(None)
    }
}

/// for numbers
pub(crate) type PtrIdxSize = (*mut usize, usize, usize);

/// for buffers
pub(crate) type PtrIdxLen = (*mut c_void, usize, usize);

// TODO: (No, invalid arg size error) Use this instead of the current KernelOptions implementation?
#[allow(dead_code)]
pub struct KernelRunner<'a, T> {
    src: &'a str,
    output: Option<Buffer<T>>,
    buf_args: Vec<PtrIdxLen>,
    num_args: Vec<PtrIdxSize>,
    gws: [usize; 3],
    lws: Option<[usize; 3]>,
    offset: Option<[usize; 3]>,
    wd: usize,
    device: CLDevice,
}

impl<'a, T: CDatatype> KernelRunner<'a, T> {
    pub fn new(
        device: &CLDevice,
        lhs: &'a mut Buffer<T>,
        gws: [usize; 3],
        src: &'a str,
    ) -> crate::Result<KernelRunner<'a, T>> {
        let wd;
        if gws[0] == 0 {
            return Err(OCLErrorKind::InvalidGlobalWorkSize.into());
        } else if gws[1] == 0 {
            wd = 1;
        } else if gws[2] == 0 {
            wd = 2;
        } else {
            wd = 3;
        }

        Ok(KernelRunner {
            src,
            output: None,
            buf_args: vec![(lhs.ptr.1, 0, lhs.len)],
            num_args: Vec::new(),
            gws,
            lws: None,
            offset: None,
            wd,
            device: device.clone(),
        })
    }

    pub fn with_lws(&mut self, lws: [usize; 3]) -> &mut KernelRunner<'a, T> {
        self.lws = Some(lws);
        self
    }

    pub fn with_offset(&mut self, offset: [usize; 3]) -> &mut Self {
        self.offset = Some(offset);
        self
    }

    /// Adds value (Matrix<U> or Buffer<U> or U) to the kernel argument list
    pub fn add_arg<U: 'a, A: KernelArg<'a, U>>(
        &'a mut self,
        arg: &'a mut A,
    ) -> &mut KernelRunner<'a, T> {
        let idx = self.buf_args.len() + self.num_args.len();

        match arg.as_number() {
            Some(number) => self.num_args.push((
                number as *const U as *mut usize,
                idx,
                core::mem::size_of::<U>(),
            )),
            None => {
                let buf = arg.some_buf().unwrap();
                self.buf_args.push((buf.ptr.1, idx, buf.len));
            }
        }
        self
    }

    /// Adds output
    pub fn with_output(&mut self, out_len: usize) -> &mut KernelRunner<'a, T> {
        self.output = Some(CLCache::get(&self.device, out_len).to_buf());
        self
    }

    /*/// Runs the kernel
    pub fn run(&mut self) -> Result<Option<Buffer<T>>, Error> {
        let kernel = CL_CACHE.with(|cache|
            cache.borrow_mut().arg_kernel_cache1(&self.device, self.src.to_string())
        )?;

        for arg in &self.buf_args {
            set_kernel_arg_ptr(&kernel, arg.1, &(arg.0 as *mut c_void), arg.2)?
        }

        for arg in &self.num_args {
            set_kernel_arg_ptr(&kernel, arg.1, &(arg.0 as *mut c_void), arg.2)?
        }

        enqueue_nd_range_kernel(&self.device.queue(), &kernel, self.wd, &self.gws, self.lws.as_ref(), self.offset)?;

        if let Some(output) = &self.output {
            // TODO: Make owned, therefore take self not ref
            return Ok(Some(output.clone()));
        }

        Ok(None)
    }*/
}

pub trait AsClCvoidPtr {
    fn as_cvoid_ptr(&self) -> *mut c_void;
    fn is_num(&self) -> bool {
        false
    }
    fn size(&self) -> usize {
        std::mem::size_of::<*mut c_void>()
    }
}

impl<T> AsClCvoidPtr for &Buffer<T> {
    fn as_cvoid_ptr(&self) -> *mut c_void {
        self.ptr.1
    }
}

impl<T> AsClCvoidPtr for Buffer<T> {
    fn as_cvoid_ptr(&self) -> *mut c_void {
        self.ptr.1
    }
}

// TODO: implement in custos-math?
/*impl<T> AsClCvoidPtr for Matrix<T> {
    fn as_cvoid_ptr(&self) -> *mut c_void {
        self.ptr.1
    }
}*/

// TODO: use this fn instead of KernelOptions
pub fn enqueue_kernel(
    device: &CLDevice,
    src: &str,
    gws: [usize; 3],
    lws: Option<[usize; 3]>,
    args: Vec<&dyn AsClCvoidPtr>,
) -> crate::Result<()> {
    let kernel = CL_CACHE.with(|cache| {
        cache
            .borrow_mut()
            .arg_kernel_cache1(device, src.to_string())
    })?;

    let wd;
    if gws[0] == 0 {
        return Err(OCLErrorKind::InvalidGlobalWorkSize.into());
    } else if gws[1] == 0 {
        wd = 1;
    } else if gws[2] == 0 {
        wd = 2;
    } else {
        wd = 3;
    }

    for (idx, arg) in args.into_iter().enumerate() {
        // TODO: IMPROVE
        set_kernel_arg_ptr(&kernel, idx, &arg.as_cvoid_ptr(), arg.size())?;
    }
    enqueue_nd_range_kernel(&device.queue(), &kernel, wd, &gws, lws.as_ref(), None)?;
    Ok(())
}
