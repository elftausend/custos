//! custos is a minimal OpenCL, CUDA and host CPU array manipulation engine / framework.
//! It provides some matrix / buffer operations: matrix multiplication (BLAS, cuBLAS), element-wise arithmetic (vector addition, ...), set all elements to zero (or default value).
//! To use more operations: [custos-math]
//!
//! [custos-math]: https://github.com/elftausend/custos-math
//!
//! ## [Examples]
//!
//! [examples]: https://github.com/elftausend/custos/tree/main/examples
//!
//! Using the host CPU as the compute device:
//!
//! [cpu_readme.rs]
//!
//! [cpu_readme.rs]: https://github.com/elftausend/custos/blob/main/examples/cpu_readme.rs
//!
//! ```rust
//! use custos::{CPU, AsDev, ClearBuf, VecRead, Buffer};
//!
//! let device = CPU::new();
//! let mut a = Buffer::from(( &device, [1, 2, 3, 4, 5, 6]));
//!     
//! // specify device for operation
//! device.clear(&mut a);
//! assert_eq!(device.read(&a), [0; 6]);
//!
//! // select() ... sets CPU as 'global device'
//! // -> when device is not specified in an operation, the 'global device' is used
//! let device = CPU::new().select();
//!
//! let mut a = Buffer::from(( &device, [1, 2, 3, 4, 5, 6]));
//! a.clear();
//!
//! assert_eq!(a.read(), vec![0; 6]);
//! ```

use std::{cell::RefCell, ffi::c_void, rc::Weak};

//pub use libs::*;
pub use buffer::*;
pub use count::*;
pub use libs::*;

use libs::cpu::InternCPU;
pub use libs::cpu::CPU;
#[cfg(feature = "cuda")]
pub use libs::cuda::{CudaDevice, InternCudaDevice};
#[cfg(feature = "opencl")]
pub use libs::opencl::{CLDevice, InternCLDevice};

pub mod libs;

mod buffer;
mod count;

pub mod number;

pub struct Error {
    pub error: Box<dyn std::error::Error + Send>,
}

impl<E: std::error::Error + PartialEq + 'static> PartialEq<E> for Error {
    fn eq(&self, other: &E) -> bool {
        let e = self.error.downcast_ref::<E>();
        if let Some(e) = e {
            return e == other;
        }
        false
    }
}

impl From<Error> for Box<dyn std::error::Error> {
    fn from(e: Error) -> Self {
        e.error
    }
}

impl Error {
    pub fn kind<E: std::error::Error + PartialEq + 'static>(&self) -> Option<&E> {
        self.error.downcast_ref::<E>()
    }
}

impl<T: std::error::Error + Send + 'static> From<T> for Error {
    fn from(error: T) -> Self {
        Error {
            error: Box::new(error),
        }
    }
}

impl core::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.error)?;
        Ok(())
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.error)?;
        Ok(())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Manage device memory
///
/// # Example
/// ```
/// use custos::{CPU, Device, Buffer, VecRead, BufFlag};
///
/// let device = CPU::new();
/// let ptrs: (*mut f32, *mut std::ffi::c_void, u64) = device.alloc(12);
///
/// let buf = Buffer {
///     ptr: ptrs,
///     len: 12,
///     flag: BufFlag::None
/// };
/// assert_eq!(vec![0.; 12], device.read(&buf));
/// ```
pub trait Device<T> {
    /// Allocate memory
    /// # Example
    /// ```
    /// use custos::{CPU, Device, Buffer, VecRead, BufFlag};
    ///
    /// let device = CPU::new();
    /// let ptrs: (*mut f32, *mut std::ffi::c_void, u64) = device.alloc(12);
    ///
    /// let buf = Buffer {
    ///     ptr: ptrs,
    ///     len: 12,
    ///     flag: BufFlag::None
    /// };
    /// assert_eq!(vec![0.; 12], device.read(&buf));
    /// ```
    fn alloc(&self, len: usize) -> (*mut T, *mut c_void, u64);

    /// Allocate new memory with data
    /// # Example
    /// ```
    /// use custos::{CPU, Device, Buffer, VecRead, BufFlag};
    ///
    /// let device = CPU::new();
    /// let ptrs: (*mut u8, *mut std::ffi::c_void, u64) = device.with_data(&[1, 5, 4, 3, 6, 9, 0, 4]);
    ///
    /// let buf = Buffer {
    ///     ptr: ptrs,
    ///     len: 8,
    ///     flag: BufFlag::None
    /// };
    /// assert_eq!(vec![1, 5, 4, 3, 6, 9, 0, 4], device.read(&buf));
    /// ```
    fn with_data(&self, data: &[T]) -> (*mut T, *mut c_void, u64);
    fn alloc_with_vec(&self, vec: Vec<T>) -> (*mut T, *mut c_void, u64) {
        self.with_data(&vec)
    }
}

///All 'base' traits?
pub trait BaseDevice<T>: Device<T> + VecRead<T> {}

pub trait ClearBuf<T> {
    /// Sets all elements of the matrix to zero.
    /// # Example
    /// ```
    /// use custos::{CPU, AsDev, ClearBuf, Buffer};
    ///
    /// let device = CPU::new().select();
    /// let mut a = Buffer::from((&device, [2, 4, 6, 8, 10, 12]));
    /// assert_eq!(a.read(), vec![2, 4, 6, 8, 10, 12]);
    ///
    /// device.clear(&mut a);
    /// assert_eq!(a.read(), vec![0; 6]);
    /// ```
    fn clear(&self, buf: &mut Buffer<T>);
}

/// Trait for reading buffers.
pub trait VecRead<T> {
    /// Read the data of a buffer into a vector
    /// # Example
    /// ```
    /// use custos::{CPU, Buffer, VecRead};
    ///
    /// let device = CPU::new();
    /// let a = Buffer::from((&device, [1., 2., 3., 3., 2., 1.,]));
    /// let read = device.read(&a);
    /// assert_eq!(vec![1., 2., 3., 3., 2., 1.,], read);
    /// ```
    fn read(&self, buf: &Buffer<T>) -> Vec<T>;
}

/// Trait for writing data to buffers.
pub trait WriteBuf<T> {
    /// Write data to the buffer.
    /// # Example
    /// ```
    /// use custos::{CPU, Buffer, WriteBuf};
    ///
    /// let device = CPU::new();
    /// let mut buf = Buffer::new(&device, 4);
    /// device.write(&mut buf, &[9, 3, 2, -4]);
    /// assert_eq!(buf.as_slice(), &[9, 3, 2, -4])
    ///
    /// ```
    fn write(&self, buf: &mut Buffer<T>, data: &[T]);
    /// Write data from <Device> Buffer to other <Device> Buffer.
    // TODO: implement, change name of fn? -> set_.. ?
    fn write_buf(&self, _dst: &mut Buffer<T>, _src: &Buffer<T>) {
        unimplemented!()
    }
}

trait ManualMem<T> {
    fn drop_buf(&self, buf: Buffer<T>);
}

pub trait CacheBuf<T> {
    #[cfg_attr(feature = "safe", doc = "```ignore")]
    /// Adds a buffer to the cache. Following calls will return this buffer, if the corresponding internal count matches with the id used in the cache.
    /// # Example
    /// ```
    /// use custos::{CPU, AsDev, VecRead, set_count, get_count, CacheBuf};
    ///
    /// let device = CPU::new().select();
    /// assert_eq!(0, get_count());
    ///
    /// let mut buf = CacheBuf::<f32>::cached_buf(&device, 10);
    /// assert_eq!(1, get_count());
    ///
    /// for value in buf.as_mut_slice() {
    ///     *value = 1.5;
    /// }
    ///    
    /// set_count(0);
    /// let buf = CacheBuf::<f32>::cached_buf(&device, 10);
    /// assert_eq!(device.read(&buf), vec![1.5; 10]);
    /// ```
    fn cached_buf(&self, len: usize) -> Buffer<T>;
}

#[derive(Debug, Clone)]
pub struct Dev {
    pub cl_device: Option<Weak<RefCell<InternCLDevice>>>,
    pub cpu: Option<Weak<RefCell<InternCPU>>>,
    pub cuda: Option<Weak<RefCell<InternCudaDevice>>>,
}

impl Dev {
    pub fn new(
        cl_device: Option<Weak<RefCell<InternCLDevice>>>,
        cpu: Option<Weak<RefCell<InternCPU>>>,
        cuda: Option<Weak<RefCell<InternCudaDevice>>>,
    ) -> Dev {
        Dev {
            cl_device,
            cpu,
            cuda,
        }
    }
}

thread_local! {
    pub static GLOBAL_DEVICE: RefCell<Dev> = RefCell::new(Dev::new(None, None, None));
}

pub trait AsDev {
    fn as_dev(&self) -> Dev;
    /// Selects self as a global device. Therefore being able to use functions for matrices without specifying a compute device.
    /// When the device is dropped, the global device is no longer available.
    ///
    /// # Example
    /// ```
    /// use custos::{CPU, VecRead, Buffer, AsDev, ClearBuf};
    ///
    /// let device = CPU::new().select();
    ///
    /// let mut a = Buffer::from((&device, vec![1.5; 5*5]));
    ///
    /// a.clear();
    ///
    /// assert_eq!(a.read(), vec![0.; 5*5]);
    /// ```
    #[must_use]
    fn select(self) -> Self
    where
        Self: AsDev + Sized,
    {
        let dev = self.as_dev();
        GLOBAL_DEVICE.with(|d| *d.borrow_mut() = dev);
        self
    }
}

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DeviceError {
    NoDeviceSelected,
}

impl DeviceError {
    pub fn as_str(&self) -> &'static str {
        match self {
            DeviceError::NoDeviceSelected => {
                "No device selected, .select() on a device was not called before get_device! call"
            }
        }
    }
}

impl core::fmt::Debug for DeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())?;
        Ok(())
    }
}

impl core::fmt::Display for DeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())?;
        Ok(())
    }
}

impl std::error::Error for DeviceError {}

#[macro_export]
/// If a device is selected, it returns the device thus giving access to the functions implemented by the trait.
/// Therfore the trait needs to be implemented for the device.
///
/// # Errors
///
/// If no device is selected, a "NoDeviceSelected" error will be returned.
///
/// # Example
/// ```
/// use custos::{Error, CPU, get_device, VecRead, AsDev, Buffer};
///
/// fn main() -> Result<(), Error> {
///     let device = CPU::new().select();
///     let read = get_device!(VecRead<f32>)?;
///
///     let buf = Buffer::from(( &device, [1.51, 6.123, 7., 5.21, 8.62, 4.765]));
///     let read = read.read(&buf);
///     assert_eq!(read, vec![1.51, 6.123, 7., 5.21, 8.62, 4.765]);
///     Ok(())
/// }
/// ```
macro_rules! get_device {
    ($t:ident<$g:ident>) => {{
        use $crate::{
            cl_dev, cpu_dev, cuda_dev, is_cl_selected, is_cpu_selected, is_cuda_selected,
            DeviceError, Error, GLOBAL_DEVICE,
        };
        let device: Result<Box<dyn $t<$g>>, Error> = GLOBAL_DEVICE.with(|device| {
            let device = device.borrow();

            let mut dev: Option<Box<dyn $t<$g>>> = None;

            if is_cpu_selected(&device) {
                dev = Some(cpu_dev(&device)?)
            }

            if is_cl_selected(&device) {
                dev = Some(cl_dev(&device)?);
            }

            if is_cuda_selected(&device) {
                dev = Some(cuda_dev(&device)?);
            }

            dev.ok_or(Error::from(DeviceError::NoDeviceSelected))
        });
        device
    }};
}

// these functions exist because: if this macro is expanded in another crate, the #[cfg(feature="...")] will not look for the feature ... in custos.

pub fn is_cuda_selected(dev: &Dev) -> bool {
    dev.cuda.is_some() && cfg!(feature = "cuda")
}

#[doc(hidden)]
#[cfg(not(feature = "cuda"))]
pub fn cuda_dev(_: &Dev) -> Result<Box<CPU>> {
    Err(Error::from(DeviceError::NoDeviceSelected))
}

#[doc(hidden)]
#[cfg(feature = "cuda")]
pub fn cuda_dev(dev: &Dev) -> Result<Box<CudaDevice>> {
    Ok(Box::new(CudaDevice::from(
        dev.cuda
            .as_ref()
            .unwrap()
            .upgrade()
            .ok_or_else(|| Error::from(DeviceError::NoDeviceSelected))?,
    )))
}

pub fn is_cl_selected(dev: &Dev) -> bool {
    dev.cl_device.is_some() && cfg!(feature = "opencl")

}

#[doc(hidden)]
#[cfg(not(feature = "opencl"))]
pub fn cl_dev(_: &Dev) -> Result<Box<CPU>> {
    Err(Error::from(DeviceError::NoDeviceSelected))
}

#[doc(hidden)]
#[cfg(feature = "opencl")]
pub fn cl_dev(dev: &Dev) -> Result<Box<CLDevice>> {
    Ok(Box::new(CLDevice::from(
        dev.cl_device
            .as_ref()
            .unwrap()
            .upgrade()
            .ok_or_else(|| Error::from(DeviceError::NoDeviceSelected))?,
    )))
}

pub fn is_cpu_selected(dev: &Dev) -> bool {
    dev.cpu.is_some()
}

#[doc(hidden)]
pub fn cpu_dev(dev: &Dev) -> Result<Box<CPU>> {
    Ok(Box::new(CPU::from(
        dev.cpu
            .as_ref()
            .unwrap()
            .upgrade()
            .ok_or_else(|| Error::from(DeviceError::NoDeviceSelected))?,
    )))
}
