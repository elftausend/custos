//! A minimal OpenCL, CUDA and host CPU array manipulation engine / framework written in Rust.
//! This crate provides the tools for executing custom array operations with the CPU, as well as with CUDA and OpenCL devices.<br>
//! This guide demonstrates how operations can be implemented for the compute devices: [implement_operations.md](implement_operations.md)<br>
//! or to see it at a larger scale, look here: [custos-math]
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
pub use devices::*;
pub use error::*;
pub use graph::*;

pub use devices::cpu::CPU;
#[cfg(feature = "cuda")]
pub use devices::cuda::CudaDevice;
#[cfg(feature = "opencl")]
pub use devices::opencl::{CLDevice, InternCLDevice};

pub mod devices;

mod buffer;
mod count;
mod error;
mod graph;

pub mod number;

/// Used to determine which device type [`Device`] is of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    CPU = 0,
    #[cfg(feature = "cuda")]
    CUDA = 1,
    #[cfg(feature = "opencl")]
    CL = 2,
    None = 3,
}

/// `Device` is another representation of a compute device.<br>
/// It stores the type of the device and a pointer to the device from which `Device` originates from.<br>
/// This is used instead of another "device" generic for [`Buffer`].
///
/// # Example
/// ```rust
/// use custos::{CPU, AsDev, Device, DeviceType};
///
/// let cpu = CPU::new();
/// let device: Device = cpu.dev();
/// assert_eq!(device.device_type, DeviceType::CPU);
/// assert_eq!(device.device as *const CPU, &cpu as *const CPU);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Device {
    pub device_type: DeviceType,
    pub device: *mut u8,
}

impl Default for Device {
    fn default() -> Self {
        Self {
            device_type: DeviceType::None,
            device: null_mut(),
        }
    }
}

thread_local! {
    pub static GLOBAL_CPU: CPU = CPU::new();
}

pub trait Device1 {
    type P;
}

/// This trait is for allocating memory on the implemented device.
///
/// # Example
/// ```
/// use custos::{CPU, Alloc, Buffer, VecRead, BufFlag, AsDev, GraphReturn};
///
/// let device = CPU::new();
/// let ptrs: (*mut f32, *mut std::ffi::c_void, u64) = device.alloc(12);
///
/// let buf = Buffer {
///     ptr: ptrs,
///     len: 12,
///     device: AsDev::dev(&device),
///     flag: BufFlag::None,
///     node: device.graph().add_leaf(12),
///     p: std::marker::PhantomData
/// };
/// assert_eq!(vec![0.; 12], device.read(&buf));
/// ```
pub trait Alloc<T> {
    /// Allocate memory on the implemented device.
    /// # Example
    /// ```
    /// use custos::{CPU, Alloc, Buffer, VecRead, BufFlag, AsDev, GraphReturn};
    ///
    /// let device = CPU::new();
    /// let ptrs: (*mut f32, *mut std::ffi::c_void, u64) = device.alloc(12);
    ///
    /// let buf = Buffer {
    ///     ptr: ptrs,
    ///     len: 12,
    ///     device: AsDev::dev(&device),
    ///     flag: BufFlag::None,
    ///     node: device.graph().add_leaf(12),
    ///     p: std::marker::PhantomData
    /// };
    /// assert_eq!(vec![0.; 12], device.read(&buf));
    /// ```
    fn alloc(&self, len: usize) -> (*mut T, *mut c_void, u64);

    /// Allocate new memory with data
    /// # Example
    /// ```
    /// use custos::{CPU, Alloc, Buffer, VecRead, BufFlag, AsDev, GraphReturn};
    ///
    /// let device = CPU::new();
    /// let ptrs: (*mut u8, *mut std::ffi::c_void, u64) = device.with_data(&[1, 5, 4, 3, 6, 9, 0, 4]);
    ///
    /// let buf = Buffer {
    ///     ptr: ptrs,
    ///     len: 8,
    ///     device: AsDev::dev(&device),
    ///     flag: BufFlag::None,
    ///     node: device.graph().add_leaf(8),
    ///     p: std::marker::PhantomData
    /// };
    /// assert_eq!(vec![1, 5, 4, 3, 6, 9, 0, 4], device.read(&buf));
    /// ```
    fn with_data(&self, data: &[T]) -> (*mut T, *mut c_void, u64)
    where
        T: Clone;

    /// If the vector `vec` was allocated previously, this function can be used in order to reduce the amount of allocations, which may be faster than using a slice of `vec`.
    fn alloc_with_vec(&self, vec: Vec<T>) -> (*mut T, *mut c_void, u64)
    where
        T: Clone,
    {
        self.with_data(&vec)
    }

    /// Creates a generic representation of the device
    fn as_dev(&self) -> Device;
}

/// Trait for implementing the clear() operation for the compute devices.
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
    /// Writes data from <Device> Buffer to other <Device> Buffer.
    // TODO: implement, change name of fn? -> set_.. ?
    fn write_buf(&self, _dst: &mut Buffer<T>, _src: &Buffer<T>) {
        unimplemented!()
    }
}

/// This trait is used to clone a buffer based on a specific device type.
pub trait CloneBuf<'a, T> {
    /// Creates a deep copy of the specified buffer.
    /// # Example
    ///
    /// ```
    /// use custos::{CPU, Buffer, CloneBuf};
    ///
    /// let device = CPU::new();
    /// let buf = Buffer::from((&device, [1., 2., 6., 2., 4.,]));
    ///
    /// let cloned = device.clone_buf(&buf);
    /// assert_eq!(buf.read(), cloned.read());
    /// ```
    fn clone_buf(&'a self, buf: &Buffer<'a, T>) -> Buffer<'a, T>;
}

/// This trait is used to retrieve a cached buffer from a specific device type.
pub trait CacheBuf<'a, T> {
    #[cfg_attr(feature = "realloc", doc = "```ignore")]
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

/// This trait is a non-generic variant for calling [`Alloc`]'s `Alloc::<T>::as_dev(..)`
pub trait AsDev {
    fn dev(&self) -> Device
    where
        Self: Alloc<u8> + Sized,
    {
        Alloc::as_dev(self)
    }
}

/// Return a device that implements the trait provided thus giving access to the functions implemented by the trait.
///
/// # Example
/// ```
/// use custos::{Error, CPU, get_device, VecRead, AsDev, Buffer};
///
/// fn main() -> Result<(), Error> {
///     let device = CPU::new();
///     let read = get_device!(device.dev(), VecRead<f32>);
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
            //&*($device.device as *mut CPU)
            match $device.device_type {
                DeviceType::CPU => &*($device.device as *mut CPU),
                #[cfg(feature="cuda")]
                DeviceType::CUDA => &*($device.device as *mut $crate::CudaDevice),
                #[cfg(feature="opencl")]
                DeviceType::CL => &*($device.device as *mut $crate::CLDevice),
                // TODO: convert to error
                _ => panic!(
                    "No device found to execute this operation with. 
                    If you are using get_device! in your own crate, 
                    you need to add 'opencl' and 'cuda' as features in your Cargo.toml."
                ),
            }

        };
        device
    }}
}
