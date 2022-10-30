use core::{ffi::c_void, fmt::Debug};

use crate::cpu::{CPUPtr, CPU};

use crate::{
    Alloc, CDatatype, CacheBuf, ClearBuf, CloneBuf, Device, DevicelessAble, Node, PtrType, VecRead,
    WriteBuf, CPUCL,
};
use alloc::vec::Vec;
pub use flag::BufFlag;
pub use num::Num;

mod flag;
mod impl_from;
mod num;

/// The underlying non-growable array structure. A `Buffer` may be encapsulated in other structs.
/// By default, the `Buffer` is a f32 CPU Buffer.
/// # Example
/// ```
/// use custos::prelude::*;
///
/// fn buffer_f32_cpu(buf: &Buffer) {}
/// fn buffer_generic<T, D: Device>(buf: &Buffer<T, D>) {}
///
/// let device = CPU::new();
/// let buf = Buffer::from((&device, [0.5, 1.3, 3.2, 2.43]));
///
/// buffer_f32_cpu(&buf);
/// buffer_generic(&buf);
/// ```
pub struct Buffer<'a, T = f32, D: Device = CPU, const N: usize = 0> {
    pub ptr: D::Ptr<T, N>,
    pub len: usize,
    pub device: Option<&'a D>,
    pub flag: BufFlag,
    pub node: Node,
}

impl<'a, T, D: Device, const N: usize> Buffer<'a, T, D, N> {
    /// Creates a zeroed (or values set to default) `Buffer` with the given length on the specified device.
    /// This `Buffer` can't outlive the device specified as a parameter.
    /// ```
    /// use custos::{CPU, Buffer};
    ///
    /// let device = CPU::new();
    /// let mut buffer = Buffer::<i32>::new(&device, 6);
    ///
    /// // this only works with CPU or unified memory buffers (this creates a slice with the host pointer)
    /// for value in &mut buffer {
    ///     *value = 2;
    /// }
    ///
    /// assert_eq!(buffer.as_slice(), &[2; 6]);
    ///
    /// ```
    pub fn new(device: &'a D, len: usize) -> Buffer<'a, T, D, N>
    where
        D: Alloc<T, N>, /*+ GraphReturn*/
    {
        let len = if N > 0 { N } else { len };
        Buffer {
            ptr: device.alloc(len),
            len,
            device: Some(device),
            flag: BufFlag::None,
            // TODO: enable, if leafs get more important
            //node: device.graph().add_leaf(len),
            node: Node::default(),
        }
    }

    /// Buffers created with this method can outlive the device used to create this `Buffer`.<br>
    /// No operations can be invoked on this `Buffer` as [`get_device!`] will panic.
    /// # Examples
    /// ```rust
    /// use custos::{CPU, Buffer};
    ///
    /// let mut buf = {
    ///     let device = CPU::new();
    ///     Buffer::<u8>::deviceless(&device, 5)
    /// };
    /// // buf.read(); // panics
    /// for (idx, element) in buf.iter_mut().enumerate() {
    ///     *element = idx as u8;
    /// }
    /// assert_eq!(buf.as_slice(), &[0, 1, 2, 3, 4]);
    /// ```
    pub fn deviceless<'b>(device: &'b D, len: usize) -> Buffer<'a, T, D, N>
    where
        D: DevicelessAble<T, N>,
    {
        Buffer {
            ptr: device.alloc(len),
            len,
            flag: BufFlag::None,
            node: Node::default(),
            device: None,
        }
    }

    #[inline]
    /// Returns all types of pointers. (host, OpenCL, CUDA)
    pub fn ptrs(&self) -> (*const T, *mut c_void, u64) {
        self.ptr.ptrs()
    }

    #[inline]
    /// Returns all types of pointers. (host, OpenCL, CUDA)
    pub fn ptrs_mut(&mut self) -> (*mut T, *mut c_void, u64) {
        self.ptr.ptrs_mut()
    }
}

impl<'a, T, D: Device> Buffer<'a, T, D> {
    pub fn device(&self) -> &'a D {
        self.device
            .expect("Called device() on a deviceless buffer.")
    }

    /// Returns the number of elements contained in `Buffer`.
    /// # Example
    /// ```
    /// use custos::{CPU, Buffer};
    ///
    /// let device = CPU::new();
    /// let a = Buffer::<i32, _>::new(&device, 10);
    /// assert_eq!(a.len(), 10)
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if `Buffer` is created without a slice.
    /// # Example
    /// ```
    /// use custos::{CPU, Buffer};
    ///
    /// let a = Buffer::<i32, ()>::from(5);
    /// assert!(a.is_empty())
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Sets all elements in `Buffer` to the default value.
    pub fn clear(&mut self)
    where
        T: CDatatype,
        D: ClearBuf<T, D>,
    {
        self.device().clear(self)
    }

    /// Reads the contents of the buffer and writes them into a vector.
    /// If it is certain whether a CPU, or an unified CPU + OpenCL Buffer, is used, calling `.as_slice()` (or deref/mut to `&/mut [&T]`) is probably preferred.
    ///
    /// # Example
    /// ```rust
    /// use custos::{CPU, Buffer};
    ///
    /// let device = CPU::new();
    /// let buf = Buffer::from((&device, [1, 2, 3, 4]));
    ///
    /// assert_eq!(buf.read(), vec![1, 2, 3, 4]);
    /// ```
    pub fn read(&self) -> Vec<T>
    where
        T: Clone + Default,
        D: VecRead<T, D>,
    {
        self.device().read(self)
    }

    /// Writes a slice to the Buffer.
    /// With a CPU buffer, the slice is just copied to the slice of the buffer.
    pub fn write(&mut self, data: &[T])
    where
        T: Clone,
        D: WriteBuf<T, D>,
    {
        self.device().write(self, data)
    }

    /// Creates a shallow copy of &self.
    ///
    /// # Safety
    /// Itself, this function does not need to be unsafe.
    /// However, declaring this function as unsafe highlights the violation of creating two or more owners for one resource.
    /// Furthermore, the resulting `Buffer` can outlive `self`.
    pub unsafe fn shallow(&self) -> Buffer<'a, T, D>
    where
        <D as Device>::Ptr<T, 0>: Copy,
    {
        Buffer {
            ptr: self.ptr,
            len: self.len,
            device: self.device,
            flag: BufFlag::Wrapper,
            node: self.node,
        }
    }

    /// Returns a shallow copy of &self, if the `realloc` feature is activated.
    /// If the `realloc` feature is not activated, it returns a deep copy / clone.
    ///
    /// # Safety
    /// Itself, this function does not need to be unsafe.
    /// However, declaring this function as unsafe highlights the violation of possibly creating two or more owners for one resource.
    /// Furthermore, the resulting `Buffer` can outlive `self`.
    pub unsafe fn shallow_or_clone(&self) -> Buffer<'a, T, D>
    where
        <D as Device>::Ptr<T, 0>: Copy,
        T: Clone,
        D: CloneBuf<'a, T>,
    {
        {
            #[cfg(not(feature = "realloc"))]
            self.shallow()
        }

        #[cfg(feature = "realloc")]
        self.clone()
    }
}

impl<'a, T> Buffer<'a, T> {
    /// Constructs a `Buffer` out of a host pointer and a length.
    /// # Example
    /// ```
    /// use custos::{Buffer, Alloc, CPU, VecRead};
    /// use std::ffi::c_void;
    ///
    /// let device = CPU::new();
    /// let ptr = device.alloc(10);
    /// let mut buf = unsafe {
    ///     Buffer::<f32>::from_raw_host(ptr.ptr, 10)
    /// };
    /// for (idx, value) in buf.iter_mut().enumerate() {
    ///     *value += idx as f32;
    /// }
    /// assert_eq!(buf.as_slice(), &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,])
    ///
    /// ```
    /// # Safety
    /// The pointer must not outlive the Buffer.
    pub unsafe fn from_raw_host(ptr: *mut T, len: usize) -> Buffer<'a, T> {
        Buffer {
            ptr: CPUPtr { ptr },
            //ptr: (ptr, null_mut(), 0),
            len,
            ..Default::default()
        }
    }
}

impl<'a, T: crate::number::Number> Buffer<'a, T, ()> {
    /// Used if the `Buffer` contains only a single value.
    ///
    /// # Example
    ///
    /// ```
    /// use custos::Buffer;
    ///
    /// let x: Buffer<f32, _> = 7f32.into();
    /// assert_eq!(x.item(), 7.);
    ///
    /// //let x: Buffer<f32> = (&mut [5., 4., 8.]).into();
    /// //assert_eq!(x.item(), 0.);
    /// ```
    pub fn item(&self) -> T
    where
        T: Default + Copy,
    {
        self.ptr.num
    }
}

#[cfg(feature = "opencl")]
impl<'a, T> Buffer<'a, T, crate::OpenCL> {
    #[inline]
    pub fn cl_ptr(&self) -> *mut c_void {
        assert!(
            !self.ptrs().1.is_null(),
            "called cl_ptr() on an invalid OpenCL buffer"
        );
        self.ptrs().1
    }
}

#[cfg(feature = "cuda")]
impl<'a, T> Buffer<'a, T, crate::CUDA> {
    // TODO: replace buf.ptr.2 with this fn, do the same with cl, cpu
    /// Returns a non null CUDA pointer
    #[inline]
    pub fn cu_ptr(&self) -> u64 {
        assert!(
            self.ptrs().2 != 0,
            "called cu_ptr() on an invalid CUDA buffer"
        );
        self.ptr.ptr
    }
}

impl<'a, T, D: CPUCL, const N: usize> Buffer<'a, T, D, N> {
    /// Returns a CPU slice. This does not work with CUDA or raw OpenCL buffers.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        D::buf_as_slice(self)
    }

    /// Returns a mutable CPU slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        D::buf_as_slice_mut(self)
    }

    /// Returns a non null host pointer
    #[inline]
    pub fn host_ptr(&self) -> *const T {
        assert!(
            !self.ptrs().0.is_null(),
            "called host_ptr() on an invalid CPU buffer"
        );
        self.ptrs().0
    }

    /// Returns a non null host pointer
    #[inline]
    pub fn host_ptr_mut(&mut self) -> *mut T {
        assert!(
            !self.ptrs().0.is_null(),
            "called host_ptr() on an invalid CPU buffer"
        );
        self.ptrs_mut().0
    }
}

impl<'a, T: Clone, D: CloneBuf<'a, T> + Device> Clone for Buffer<'a, T, D> {
    fn clone(&self) -> Self {
        //get_device!(self.device, CloneBuf<T>).clone_buf(self)
        self.device().clone_buf(self)
    }
}

/*#[cfg(feature = "safe")]
unsafe impl<T> Send for Buffer<'a, T> {}
#[cfg(feature = "safe")]
unsafe impl<T> Sync for Buffer<'a, T> {}*/

impl<T, D: Device, const N: usize> Drop for Buffer<'_, T, D, N> {
    fn drop(&mut self) {
        if self.flag != BufFlag::None {
            return;
        }

        unsafe {
            self.ptr.dealloc(self.len);
        }
    }
}

impl<'a, T, D: Device, const N: usize> Default for Buffer<'a, T, D, N>
where
    D::Ptr<T, N>: Default,
{
    fn default() -> Self {
        Self {
            ptr: D::Ptr::<T, N>::default(),
            flag: BufFlag::default(),
            len: Default::default(),
            device: None,
            node: Node::default(),
        }
    }
}

impl<T, D: CPUCL> AsRef<[T]> for Buffer<'_, T, D> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, D: CPUCL> AsMut<[T]> for Buffer<'_, T, D> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

/// A `Buffer` dereferences into a slice.
///
/// # Examples
///
/// ```
/// use custos::{Buffer, CPU};
///
/// let device = CPU::new();
///
/// let a = Buffer::from((&device, [1., 2., 3., 4.,]));
/// let b = Buffer::from((&device, [2., 3., 4., 5.,]));
///
/// let mut c = Buffer::from((&device, [0.; 4]));
///
/// let slice_add = |a: &[f64], b: &[f64], c: &mut [f64]| {
///     for i in 0..c.len() {
///         c[i] = a[i] + b[i];
///     }
/// };
///
/// slice_add(&a, &b, &mut c);
/// assert_eq!(c.as_slice(), &[3., 5., 7., 9.,]);
/// ```
impl<const N: usize, T, D: CPUCL> core::ops::Deref for Buffer<'_, T, D, N> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// A `Buffer` dereferences into a slice.
///
/// # Examples
///
/// ```
/// use custos::{Buffer, CPU};
///  
/// let device = CPU::new();
///
/// let a = Buffer::from((&device, [4., 2., 3., 4.,]));
/// let b = Buffer::from((&device, [2., 3., 6., 5.,]));
/// let mut c = Buffer::from((&device, [0.; 4]));
///
/// let slice_add = |a: &[f64], b: &[f64], c: &mut [f64]| {
///     for i in 0..c.len() {
///         c[i] = a[i] + b[i];
///     }
/// };
/// slice_add(&a, &b, &mut c);
/// assert_eq!(c.as_slice(), &[6., 5., 9., 9.,]);
/// ```
impl<const N: usize, T, D: CPUCL> core::ops::DerefMut for Buffer<'_, T, D, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T, D> Debug for Buffer<'_, T, D>
where
    T: Debug + Default + Copy,
    D: VecRead<T, D> + Device,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Buffer")
            .field("ptr (CPU, CL, CU)", &self.ptrs())
            .field("len", &self.len);
        writeln!(f, ",")?;

        if !self.ptrs().0.is_null() {
            writeln!(f, "CPU:    {:?}", self.device().read(self))?;
        }

        #[cfg(feature = "opencl")]
        if !self.ptrs().1.is_null() {
            write!(f, "OpenCL: {:?}, ", self.device().read(self))?;
        }

        #[cfg(feature = "cuda")]
        if self.ptrs().2 != 0 {
            write!(f, "CUDA: {:?}, ", self.device().read(self))?;
        }

        write!(
            f,
            "datatype={}, device={device:?} }}",
            core::any::type_name::<T>(),
            device = core::any::type_name::<D>()
        )
    }
}

impl<'a, T, D: CPUCL> core::iter::IntoIterator for &'a Buffer<'_, T, D> {
    type Item = &'a T;

    type IntoIter = alloc::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, D: CPUCL> core::iter::IntoIterator for &'a mut Buffer<'_, T, D> {
    type Item = &'a mut T;

    type IntoIter = core::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// Adds a `Buffer` to the "cache chain".
/// Following calls will return this `Buffer`,
/// if the corresponding internal count matches with the id used in the cache.
///
///
/// # Example
#[cfg_attr(feature = "realloc", doc = "```ignore")]
#[cfg_attr(not(feature = "realloc"), doc = "```")]
/// use custos::{CPU, cached, VecRead, set_count, get_count};
///
/// let device = CPU::new();
/// assert_eq!(0, get_count());
///
/// let mut buf = cached::<f32, _>(&device, 10);
/// assert_eq!(1, get_count());
///
/// for value in buf.as_mut_slice() {
///     *value = 1.5;
/// }
///    
/// let new_buf = cached::<i32, _>(&device, 10);
/// assert_eq!(2, get_count());
///
/// set_count(0);
/// let buf = cached::<f32, _>(&device, 10);
/// assert_eq!(device.read(&buf), vec![1.5; 10]);
/// ```
pub fn cached<'a, T, D: CacheBuf<'a, T> + Device>(device: &'a D, len: usize) -> Buffer<'a, T, D> {
    device.cached(len)
}
