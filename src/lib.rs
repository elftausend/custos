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
//! use custos::{CPU, ClearBuf, VecRead, Buffer};
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
//! let device = CPU::new();
//!
//! let mut a = Buffer::from(( &device, [1, 2, 3, 4, 5, 6]));
//! a.clear();
//!
//! assert_eq!(a.read(), vec![0; 6]);
//! ```
use std::{ffi::c_void, ptr::null_mut};

//pub use libs::*;
pub use buffer::*;
pub use count::*;
pub use libs::*;

pub use libs::cpu::CPU;
#[cfg(feature = "cuda")]
pub use libs::cuda::{CudaDevice, InternCudaDevice};
#[cfg(feature = "opencl")]
pub use libs::opencl::{CLDevice, InternCLDevice};

pub mod libs;

mod buffer;
mod count;

pub mod number;

#[derive(Debug, Clone, Copy)]
pub enum DeviceType {
    CPU = 0,
    #[cfg(feature="cuda")]
    CUDA = 1,
    #[cfg(feature="opencl")]
    CL = 2,
    None = 3,
}

#[derive(Debug, Clone, Copy)]
pub struct Device {
    pub device_type: DeviceType,
    pub device: *mut u8
}

impl Default for Device {
    fn default() -> Self {
        Self { device_type: DeviceType::None, device: null_mut() }
    }
}

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
/// use custos::{CPU, Alloc, Buffer, VecRead, BufFlag, AsDev};
///
/// let device = CPU::new();
/// let ptrs: (*mut f32, *mut std::ffi::c_void, u64) = device.alloc(12);
///
/// let buf = Buffer {
///     ptr: ptrs,
///     len: 12,
///     device: AsDev::as_dev(&device),
///     flag: BufFlag::None,
///     p: std::marker::PhantomData
/// };
/// assert_eq!(vec![0.; 12], device.read(&buf));
/// ```
pub trait Alloc<T> {
    /// Allocate memory
    /// # Example
    /// ```
    /// use custos::{CPU, Alloc, Buffer, VecRead, BufFlag, AsDev};
    ///
    /// let device = CPU::new();
    /// let ptrs: (*mut f32, *mut std::ffi::c_void, u64) = device.alloc(12);
    ///
    /// let buf = Buffer {
    ///     ptr: ptrs,
    ///     len: 12,
    ///     device: AsDev::as_dev(&device),
    ///     flag: BufFlag::None,
    ///     p: std::marker::PhantomData
    /// };
    /// assert_eq!(vec![0.; 12], device.read(&buf));
    /// ```
    fn alloc(&self, len: usize) -> (*mut T, *mut c_void, u64);

    /// Allocate new memory with data
    /// # Example
    /// ```
    /// use custos::{CPU, Alloc, Buffer, VecRead, BufFlag, AsDev};
    ///
    /// let device = CPU::new();
    /// let ptrs: (*mut u8, *mut std::ffi::c_void, u64) = device.with_data(&[1, 5, 4, 3, 6, 9, 0, 4]);
    ///
    /// let buf = Buffer {
    ///     ptr: ptrs,
    ///     len: 8,
    ///     device: AsDev::as_dev(&device),
    ///     flag: BufFlag::None,
    ///     p: std::marker::PhantomData
    /// };
    /// assert_eq!(vec![1, 5, 4, 3, 6, 9, 0, 4], device.read(&buf));
    /// ```
    fn with_data(&self, data: &[T]) -> (*mut T, *mut c_void, u64);
    fn alloc_with_vec(&self, vec: Vec<T>) -> (*mut T, *mut c_void, u64) {
        self.with_data(&vec)
    }
    fn as_dev(&self) -> Device;
}

///All 'base' traits?
pub trait BaseDevice<T>: Alloc<T> + VecRead<T> {}

pub trait ClearBuf<T> {
    /// Sets all elements of the matrix to zero.
    /// # Example
    /// ```
    /// use custos::{CPU, ClearBuf, Buffer};
    ///
    /// let device = CPU::new();
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

pub trait CacheBuf<'a, T> {
    /// Adds a buffer to the cache. Following calls will return this buffer, if the corresponding internal count matches with the id used in the cache.
    /// # Example
    /// ```
    /// use custos::{CPU, VecRead, set_count, get_count, CacheBuf};
    ///
    /// let device = CPU::new();
    /// assert_eq!(0, get_count());
    ///
    /// let mut buf = CacheBuf::<f32>::cached(&device, 10);
    /// assert_eq!(1, get_count());
    ///
    /// for value in buf.as_mut_slice() {
    ///     *value = 1.5;
    /// }
    ///    
    /// set_count(0);
    /// let buf = CacheBuf::<f32>::cached(&device, 10);
    /// assert_eq!(device.read(&buf), vec![1.5; 10]);
    /// ```
    fn cached(&'a self, len: usize) -> Buffer<'a, T>;
}

pub trait AsDev {
    fn as_dev(&self) -> Device 
    where
        Self: Alloc<u8> + Sized,
    {
        Alloc::as_dev(self)
    }
    
}

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DeviceError {
    NoDeviceSelected,
    ConstructError,
}

impl DeviceError {
    pub fn as_str(&self) -> &'static str {
        match self {
            DeviceError::NoDeviceSelected => {
                "No device selected, .select() on a device was not called before get_device! call"
            }
            DeviceError::ConstructError => {
                "Only a non-drop buffer can be converted to a CPU+OpenCL buffer"
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

/// Return a device that implements the trait provided thus giving access to the functions implemented by the trait.
///
/// # Example
/// ```
/// use custos::{Error, CPU, get_device, VecRead, AsDev, Buffer};
///
/// fn main() -> Result<(), Error> {
///     let device = CPU::new();
///     let read = get_device!(device.as_dev(), VecRead<f32>);
///
///     let buf = Buffer::from(( &device, [1.51, 6.123, 7., 5.21, 8.62, 4.765]));
///     let read = read.read(&buf);
///     assert_eq!(read, vec![1.51, 6.123, 7., 5.21, 8.62, 4.765]);
///     Ok(())
/// }
/// ```
#[macro_export]
macro_rules! get_device {
    ($device:expr, $t:ident<$g:ident>) => {{
        use $crate::{ DeviceType, CPU };

        let device: &dyn $t<$g> = unsafe {
            match $device.device_type {
                DeviceType::CPU => &*($device.device as *mut CPU),
                #[cfg(feature="cuda")]
                DeviceType::CUDA => &*($device.device as *mut $crate::CudaDevice),
                #[cfg(feature="opencl")]
                DeviceType::CL => &*($device.device as *mut $crate::CLDevice),
                // TODO: convert to error
                DeviceType::None => panic!("No device found to execute this operation with."),
            }
        };
        device
    }}
}
