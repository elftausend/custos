use std::{ffi::c_void, fmt::Debug, ptr::null_mut};

#[cfg(feature = "opencl")]
#[cfg(feature = "safe")]
use crate::opencl::api::{release_mem_object, retain_mem_object};
use crate::{get_device, CDatatype, CacheBuf, ClearBuf, Device, VecRead, WriteBuf};

#[cfg(not(feature = "safe"))]
use crate::number::Number;

#[cfg_attr(not(feature = "safe"), derive(Clone, Copy))]
pub struct Buffer<T> {
    pub ptr: (*mut T, *mut c_void, u64),
    pub len: usize,
}

impl<T> Buffer<T> {
    /// Creates a zeroed (or values set to default) buffer with the given length on the specified device.
    /// ```
    /// use custos::{CPU, Buffer};
    ///
    /// let device = CPU::new();
    /// let mut buffer = Buffer::<i32>::new(&device, 6);
    ///
    /// // this works only with cpu buffers (this creates a slice with the host pointer)
    /// for value in &mut buffer {
    ///     *value = 2;
    /// }
    /// 
    /// assert_eq!(buffer.as_slice(), &[2; 6]);
    ///
    /// ```
    pub fn new<D: Device<T>>(device: &D, len: usize) -> Buffer<T> {
        Buffer {
            ptr: device.alloc(len),
            len,
        }
    }

    /// Constructs a Buffer with a host pointer and a length.
    /// # Example
    /// ```
    /// use custos::{Buffer, Device, CPU, VecRead};
    /// use std::ffi::c_void;
    /// 
    /// let device = CPU::new();
    /// let ptrs: (*mut f32, *mut c_void, u64) = device.alloc(10);
    /// let mut buf = unsafe {
    ///     Buffer::from_raw_host(ptrs.0, 10)
    /// };
    /// for (idx, value) in buf.iter_mut().enumerate() {
    ///     *value += idx as f32;
    /// }
    /// assert_eq!(buf.as_slice(), &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,])
    /// ```
    pub unsafe fn from_raw_host(ptr: *mut T, len: usize) -> Buffer<T> {
        Buffer {
            ptr: (ptr, null_mut(), 0),
            len
        }
    }

    /// Returns the number of elements contained in the buffer.
    /// # Example
    /// ```
    /// use custos::{CPU, Buffer};
    ///
    /// let device = CPU::new();
    /// let a = Buffer::<i32>::new(&device, 10);
    /// assert_eq!(a.len(), 10)
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    #[cfg_attr(feature = "safe", doc = "```ignore")]
    /// Returns `true` if the buffer is created without a slice.
    /// # Example
    /// ```
    /// use custos::{CPU, Buffer};
    ///
    /// let device = CPU::new();
    /// let a = Buffer::<i32>::from(5);
    /// assert!(a.is_empty())
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a non null host pointer
    pub fn host_ptr(&self) -> *mut T {
        assert!(!self.ptr.0.is_null(), "");
        self.ptr.0
    }

    // TODO: replace buf.ptr.2 with this fn, do the same with cl, cpu
    /// Returns a non null CUDA pointer
    pub fn cu_ptr(&self) -> u64 {
        assert!(self.ptr.2 != 0, "");
        self.ptr.2
    }

    /// Returns a CPU slice. This does not work with CUDA or OpenCL buffers.
    pub fn as_slice(&self) -> &[T] {
        assert!(
            !self.ptr.0.is_null(),
            "called as_slice() on a non CPU buffer (this would dereference a null pointer)"
        );
        unsafe { std::slice::from_raw_parts(self.ptr.0, self.len) }
    }

    /// Returns a mutable CPU slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        assert!(
            !self.ptr.0.is_null(),
            "called as_mut_slice() on a non CPU buffer (this would dereference a null pointer)"
        );
        unsafe { std::slice::from_raw_parts_mut(self.ptr.0, self.len) }
    }

    #[cfg(not(feature = "safe"))]
    /// Used if the buffer contains only a single value.
    ///
    /// # Example
    ///
    /// ```
    /// use custos::Buffer;
    ///
    /// let x: Buffer<f32> = 7f32.into();
    /// assert_eq!(x.item(), 7.);
    ///
    /// //let x: Buffer<f32> = (&mut [5., 4., 8.]).into();
    /// //assert_eq!(x.item(), 0.);
    /// ```
    pub fn item(&self) -> T
    where
        T: Default + Copy,
    {
        if self.len == 0 {
            return unsafe { *self.ptr.0 };
        }
        T::default()
    }

    pub fn clear(&mut self)
    where
        T: CDatatype,
    {
        let device = get_device!(ClearBuf<T>).unwrap();
        device.clear(self)
    }

    /// Reads the contents of the buffer into a vector.
    /// If it is certain whether a CPU, or an unified CPU + OpenCL Buffer, is used, calling `.as_slice()` (or deref/mut to `&/mut [&T]`) is probably preferred.
    pub fn read(&self) -> Vec<T>
    where
        T: Copy + Default,
    {
        let device = get_device!(VecRead<T>).unwrap();
        device.read(self)
    }

    /// Writes a slice to the vector.
    /// With a CPU buffer, the slice is just copied to the slice of the buffer.
    pub fn write(&mut self, data: &[T]) 
    where
        T: Copy
    {
        get_device!(WriteBuf<T>).unwrap().write(self, data)
    }
}

#[cfg(feature = "safe")]
unsafe impl<T> Send for Buffer<T> {}
#[cfg(feature = "safe")]
unsafe impl<T> Sync for Buffer<T> {}

// TODO: Safe mode and cuda clone | cuda ptr reference counted?
#[cfg(feature = "safe")]
impl<T: Clone> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        if !self.ptr.0.is_null() && self.ptr.1.is_null() {
            let mut ptr = self.ptr;
            ptr.0 = Box::into_raw(self.as_slice().to_vec().into_boxed_slice()) as *mut T;
            return Self { ptr, len: self.len };
        }

        #[cfg(feature = "opencl")]
        if !self.ptr.1.is_null() {
            retain_mem_object(self.ptr.1).unwrap();
        }

        #[cfg(feature = "cuda")]
        if self.ptr.2 != 0 {
            unimplemented!("At the moment, cloning a CUDA Buffer is undefined");
        };

        Self {
            ptr: self.ptr,
            len: self.len,
        }
    }
}

impl<A: Clone + Default> FromIterator<A> for Buffer<A> {
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        let device = get_device!(Device<A>).unwrap();
        let from_iter = Vec::from_iter(iter);
        Buffer {
            len: from_iter.len(),
            ptr: device.alloc_with_vec(from_iter),
        }        
    }
}

#[cfg(feature = "safe")]
impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.0.is_null() && self.ptr.1.is_null() {
                let ptr = std::slice::from_raw_parts_mut(self.ptr.0, self.len);
                Box::from_raw(ptr);
            }

            #[cfg(feature = "opencl")]
            if !self.ptr.1.is_null() {
                release_mem_object(self.ptr.1).unwrap()
            }

            #[cfg(feature = "cuda")]
            if self.ptr.2 != 0 {
                use crate::cuda::api::cufree;
                cufree(self.ptr.2).unwrap();
            }
        }
    }
}

impl<T> Default for Buffer<T> {
    fn default() -> Self {
        Self {
            ptr: (null_mut(), null_mut(), 0),
            len: Default::default(),
        }
    }
}

impl<T> AsRef<[T]> for Buffer<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for Buffer<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

/// A buffer dereferences into a slice.
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
impl<T> std::ops::Deref for Buffer<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// A buffer dereferences into a slice.
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
impl<T> std::ops::DerefMut for Buffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T: Debug + Default + Copy> Debug for Buffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("ptr (CPU, CL, CU)", &self.ptr)
            .field("len", &self.len);
        writeln!(f, ",")?;
        if !self.ptr.0.is_null() {
            writeln!(f, "CPU:    {:?}", self.as_slice())?;
        }

        #[cfg(feature = "opencl")]
        if !self.ptr.1.is_null() {
            let read = get_device!(VecRead<T>).unwrap();
            write!(f, "OpenCL: {:?}, ", read.read(self))?;
        }

        #[cfg(feature = "cuda")]
        if self.ptr.2 != 0 {
            let read = get_device!(VecRead<T>).unwrap();
            write!(f, "CUDA: {:?}, ", read.read(self))?;
        }

        write!(f, "datatype={} }}", std::any::type_name::<T>())
    }
}

/*
pub struct IntoIter<T> {
    _ptr: *mut T,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl<T> std::iter::IntoIterator for Buffer<T> {
    type Item = T;

    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        assert!(self.ptr.0 != std::ptr::null_mut(), "called as_slice() on a non CPU buffer (this would dereference a null pointer)");
        unsafe {
            let ptr = self.ptr.0;
            let _end = if core::mem::size_of::<T>() == 0 {
                (ptr as *const i8).wrapping_offset(self.len() as isize) as *const T

            } else {
                ptr.add(self.len()) as *const T
            };

            IntoIter {
                _ptr: ptr
            }
        }
    }
}
*/

impl<'a, T> std::iter::IntoIterator for &'a Buffer<T> {
    type Item = &'a T;

    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> std::iter::IntoIterator for &'a mut Buffer<T> {
    type Item = &'a mut T;

    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(not(feature = "safe"))]
impl<T: Number> From<T> for Buffer<T> {
    fn from(val: T) -> Self {
        Buffer {
            ptr: (Box::into_raw(Box::new(val)), null_mut(), 0),
            len: 0,
        }
    }
}

impl<T: Clone, const N: usize> From<(&Box<dyn Device<T>>, &[T; N])> for Buffer<T> {
    fn from(device_slice: (&Box<dyn Device<T>>, &[T; N])) -> Self {
        Buffer {
            ptr: device_slice.0.with_data(device_slice.1),
            len: device_slice.1.len(),
        }
    }
}

impl<T: Clone> From<(&Box<dyn Device<T>>, usize)> for Buffer<T> {
    fn from(device_len: (&Box<dyn Device<T>>, usize)) -> Self {
        Buffer {
            ptr: device_len.0.alloc(device_len.1),
            len: device_len.1,
        }
    }
}

impl<T: Clone, D: Device<T>, const N: usize> From<(&D, [T; N])> for Buffer<T> {
    fn from(device_slice: (&D, [T; N])) -> Self {
        Buffer {
            ptr: device_slice.0.with_data(&device_slice.1),
            len: device_slice.1.len(),
        }
    }
}

impl<T: Clone, D: Device<T>> From<(&D, &[T])> for Buffer<T> {
    fn from(device_slice: (&D, &[T])) -> Self {
        Buffer {
            ptr: device_slice.0.with_data(device_slice.1),
            len: device_slice.1.len(),
        }
    }
}

impl<T: Clone, D: Device<T>> From<(&D, Vec<T>)> for Buffer<T> {
    fn from(device_slice: (&D, Vec<T>)) -> Self {
        let len = device_slice.1.len();
        Buffer {
            ptr: device_slice.0.alloc_with_vec(device_slice.1),
            len,
        }
    }
}

impl<T: Clone> From<(Box<dyn Device<T>>, Vec<T>)> for Buffer<T> {
    fn from(device_slice: (Box<dyn Device<T>>, Vec<T>)) -> Self {
        let len = device_slice.1.len();
        Buffer {
            ptr: device_slice.0.alloc_with_vec(device_slice.1),
            len,
        }
    }
}

impl<T: Clone, D: Device<T>> From<(&D, &Vec<T>)> for Buffer<T> {
    fn from(device_slice: (&D, &Vec<T>)) -> Self {
        let len = device_slice.1.len();
        Buffer {
            ptr: device_slice.0.with_data(device_slice.1),
            len,
        }
    }
}

// TODO: unsafe from raw parts fn?
#[cfg(not(feature = "safe"))]
impl<T: Copy> From<(*mut T, usize)> for Buffer<T> {
    fn from(info: (*mut T, usize)) -> Self {
        Buffer {
            ptr: (info.0, null_mut(), 0),
            len: info.1,
        }
    }
}

impl<T: CDatatype> From<(*mut c_void, usize)> for Buffer<T> {
    fn from(info: (*mut c_void, usize)) -> Self {
        Buffer {
            ptr: (null_mut(), info.0, 0),
            len: info.1,
        }
    }
}

#[cfg_attr(feature = "safe", doc = "```ignore")]
/// Adds a buffer to the "cache chain".
/// Following calls will return this buffer,
/// if the corresponding internal count matches with the id used in the cache.
///
/// # 'safe' feature
/// An zeroed buffer with the specified len is returned.
///
/// # Example
/// ```
/// use custos::{CPU, AsDev, cached, VecRead, set_count, get_count};
///
/// let device = CPU::new().select();
/// assert_eq!(0, get_count());
///
/// let mut buf = cached::<f32>(10);
/// assert_eq!(1, get_count());
///
/// for value in buf.as_mut_slice() {
///     *value = 1.5;
/// }
///    
/// let new_buf = cached::<i32>(10);
/// assert_eq!(2, get_count());
///
/// set_count(0);
/// let buf = cached::<f32>(10);
/// assert_eq!(device.read(&buf), vec![1.5; 10]);
/// ```
pub fn cached<T: Default+Copy>(len: usize) -> Buffer<T> {
    let device = get_device!(CacheBuf<T>).unwrap();
    device.cached_buf(len)
}
