use std::alloc::Layout;
use std::marker::PhantomData;
use std::{ffi::c_void, fmt::Debug, ptr::null_mut};

#[cfg(feature = "opencl")]
use crate::opencl::api::release_mem_object;
use crate::{get_device, CDatatype, ClearBuf, Alloc, VecRead, WriteBuf, Device, CacheBuf, GLOBAL_CPU, AsDev};

use crate::number::Number;

#[derive(Debug, Clone)]
pub enum BufFlag {
    None,
    Cache,
    Wrapper,
}

impl PartialEq for BufFlag {
    fn eq(&self, other: &Self) -> bool {
        core::mem::discriminant(self) == core::mem::discriminant(other)
    }
}

pub struct Buffer<'a, T> {
    pub ptr: (*mut T, *mut c_void, u64),
    pub len: usize,
    pub device: Device,
    pub flag: BufFlag,
    pub p: PhantomData<&'a T>,
}

impl<'a, T> Buffer<'a, T> {
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
    pub fn new<'b, D: Alloc<T> + ?Sized>(device: &'b D, len: usize) -> Buffer<'a, T> {
        Buffer {
            ptr: device.alloc(len),
            len,
            device: device.as_dev(),
            flag: BufFlag::None,
            p: PhantomData,
            
        }
    }

    /// Constructs a Buffer with a host pointer and a length.
    /// # Example
    /// ```
    /// use custos::{Buffer, Alloc, CPU, VecRead};
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
    pub unsafe fn from_raw_host(ptr: *mut T, len: usize) -> Buffer<'a, T> {
        Buffer {
            ptr: (ptr, null_mut(), 0),
            len,
            device: Device::default(),
            flag: BufFlag::None,
            p: PhantomData,
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
    #[inline]
    pub fn host_ptr(&self) -> *mut T {
        assert!(
            !self.ptr.0.is_null(),
            "called host_ptr() on an invalid CPU buffer"
        );
        self.ptr.0
    }

    #[cfg(feature = "opencl")]
    #[inline]
    pub fn cl_ptr(&self) -> *mut c_void {
        assert!(
            !self.ptr.1.is_null(),
            "called cl_ptr() on an invalid OpenCL buffer"
        );
        self.ptr.1
    }

    // TODO: replace buf.ptr.2 with this fn, do the same with cl, cpu
    /// Returns a non null CUDA pointer
    #[cfg(feature = "cuda")]
    #[inline]
    pub fn cu_ptr(&self) -> u64 {
        assert!(
            self.ptr.2 != 0,
            "called cu_ptr() on an invalid CUDA buffer"
        );
        self.ptr.2
    }

    /// Returns a CPU slice. This does not work with CUDA or OpenCL buffers.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        assert!(
            /*self.flag == BufFlag::Wrapper ||*/
            !self.ptr.0.is_null(), 
            "called as_slice() on an invalid CPU buffer (this would dereference an invalid pointer)"
        );
        unsafe { std::slice::from_raw_parts(self.ptr.0, self.len) }
    }

    /// Returns a mutable CPU slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        assert!(
            //self.flag == BufFlag::Wrapper
            !self.ptr.0.is_null(),
            "called as_mut_slice() on a non CPU buffer (this would dereference a null pointer)"
        );
        unsafe { std::slice::from_raw_parts_mut(self.ptr.0, self.len) }
    }

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
        let device = get_device!(self.device, ClearBuf<T>);
        device.clear(self)
    }

    /// Reads the contents of the buffer into a vector.
    /// If it is certain whether a CPU, or an unified CPU + OpenCL Buffer, is used, calling `.as_slice()` (or deref/mut to `&/mut [&T]`) is probably preferred.
    pub fn read(&self) -> Vec<T>
    where
        T: Copy + Default,
    {
        let device = get_device!(self.device, VecRead<T>);
        device.read(self)
    }

    /// Writes a slice to the vector.
    /// With a CPU buffer, the slice is just copied to the slice of the buffer.
    pub fn write(&mut self, data: &[T])
    where
        T: Copy,
    {
        get_device!(self.device, WriteBuf<T>).write(self, data)
    }

    #[cfg(feature="cuda")]
    pub fn to_cuda<'c>(&self, cuda_device: &'c crate::CudaDevice) -> crate::Result<Buffer<'c, T>> {
        use crate::{DeviceType, DeviceError};

        if self.device.device_type != DeviceType::CPU {
            return Err(DeviceError::CPUtoCUDA.into());
        }

        let mut out = crate::Cache::get(cuda_device, self.len);
        cuda_device.write(&mut out, self);
        Ok(out)
    }
}

/*#[cfg(feature = "safe")]
unsafe impl<T> Send for Buffer<'a, T> {}
#[cfg(feature = "safe")]
unsafe impl<T> Sync for Buffer<'a, T> {}*/

impl<T> Clone for Buffer<'_, T> {
    fn clone(&self) -> Self {
        assert!(self.flag == BufFlag::Cache, "Called .clone() on a non-cache buffer. Use a reference counted approach instead.");
    
        Self {
            ptr: self.ptr,
            len: self.len,
            device: self.device,
            flag: self.flag.clone(),
            p: PhantomData,
        }
    }
}


impl<A: Clone + Default> FromIterator<A> for Buffer<'_, A> {
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        GLOBAL_CPU.with(|device| {
            let from_iter = Vec::from_iter(iter);
    
            Buffer {
                len: from_iter.len(),
                ptr: device.alloc_with_vec(from_iter),
                device: device.dev(),
                flag: BufFlag::None,
                p: PhantomData,
            }
        })
    }
}

impl<T> Drop for Buffer<'_, T> {
    fn drop(&mut self) {
        if self.flag != BufFlag::None {
            return;
        }
    
        unsafe {
            if !self.ptr.0.is_null() && self.ptr.1.is_null() {
                let layout = Layout::array::<T>(self.len).unwrap();
                std::alloc::dealloc(self.ptr.0 as *mut u8, layout);
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

impl<T> Default for Buffer<'_, T> {
    fn default() -> Self {
        Self {
            ptr: (null_mut(), null_mut(), 0),
            len: Default::default(),
            device: Default::default(),
            flag: BufFlag::None,
            p: PhantomData,
        }
    }
}

impl<T> AsRef<[T]> for Buffer<'_, T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for Buffer<'_, T> {
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
impl<T> std::ops::Deref for Buffer<'_, T> {
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
impl<T> std::ops::DerefMut for Buffer<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T: Debug + Default + Copy> Debug for Buffer<'_, T> {
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
            let read = get_device!(self.device, VecRead<T>);
            write!(f, "OpenCL: {:?}, ", read.read(self))?;
        }

        #[cfg(feature = "cuda")]
        if self.ptr.2 != 0 {
            let read = get_device!(self.device, VecRead<T>);
            write!(f, "CUDA: {:?}, ", read.read(self))?;
        }

        write!(f, "datatype={}, device={device:?} }}", std::any::type_name::<T>(), device=self.device.device_type)
    }
}

impl<'a, T> std::iter::IntoIterator for &'a Buffer<'_, T> {
    type Item = &'a T;

    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> std::iter::IntoIterator for &'a mut Buffer<'_, T> {
    type Item = &'a mut T;

    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

// TODO: test if working correctly (no safe mode)
// mind deallocation
impl<T: Number> From<T> for Buffer<'_, T> {
    fn from(val: T) -> Self {
        Buffer {
            ptr: (Box::into_raw(Box::new(val)), null_mut(), 0),
            len: 0,
            device: Device::default(),
            flag: BufFlag::None,
            p: PhantomData,
        }
    }
}

impl<T: Clone, const N: usize> From<(&Box<dyn Alloc<T>>, &[T; N])> for Buffer<'_, T> {
    fn from(device_slice: (&Box<dyn Alloc<T>>, &[T; N])) -> Self {
        Buffer {
            ptr: device_slice.0.with_data(device_slice.1),
            len: device_slice.1.len(),
            device: device_slice.0.as_dev(),
            flag: BufFlag::None,
            p: PhantomData,
        }
    }
}

impl<T: Clone> From<(&Box<dyn Alloc<T>>, usize)> for Buffer<'_, T> {
    fn from(device_len: (&Box<dyn Alloc<T>>, usize)) -> Self {
        Buffer {
            ptr: device_len.0.alloc(device_len.1),
            len: device_len.1,
            device: device_len.0.as_dev(),
            flag: BufFlag::None,
            p: PhantomData,
        }
    }
}

impl<T: Clone, D: Alloc<T>, const N: usize> From<(&D, [T; N])> for Buffer<'_, T> {
    fn from(device_slice: (&D, [T; N])) -> Self {
        Buffer {
            ptr: device_slice.0.with_data(&device_slice.1),
            len: device_slice.1.len(),
            device: device_slice.0.as_dev(),
            flag: BufFlag::None,
            p: PhantomData,
        }
    }
}

impl<T: Clone, D: Alloc<T>> From<(&D, &[T])> for Buffer<'_, T> {
    fn from(device_slice: (&D, &[T])) -> Self {
        Buffer {
            ptr: device_slice.0.with_data(device_slice.1),
            len: device_slice.1.len(),
            device: device_slice.0.as_dev(),
            flag: BufFlag::None,
            p: PhantomData,
        }
    }
}

impl<T: Clone, D: Alloc<T>> From<(&D, Vec<T>)> for Buffer<'_, T> {
    fn from(device_slice: (&D, Vec<T>)) -> Self {
        let len = device_slice.1.len();
        Buffer {
            ptr: device_slice.0.alloc_with_vec(device_slice.1),
            len,
            device: device_slice.0.as_dev(),
            flag: BufFlag::None,
            p: PhantomData,
        }
    }
}

impl<T: Clone> From<(Box<dyn Alloc<T>>, Vec<T>)> for Buffer<'_, T> {
    fn from(device_slice: (Box<dyn Alloc<T>>, Vec<T>)) -> Self {
        let len = device_slice.1.len();
        Buffer {
            ptr: device_slice.0.alloc_with_vec(device_slice.1),
            len,
            device: device_slice.0.as_dev(),
            flag: BufFlag::None,
            p: PhantomData,
        }
    }
}

impl<T: Clone, D: Alloc<T>> From<(&D, &Vec<T>)> for Buffer<'_, T> {
    fn from(device_slice: (&D, &Vec<T>)) -> Self {
        let len = device_slice.1.len();
        Buffer {
            ptr: device_slice.0.with_data(device_slice.1),
            len,
            device: device_slice.0.as_dev(),
            flag: BufFlag::None,
            p: PhantomData,
        }
    }
}

// TODO: check if Wrapper flag fits
// TODO: unsafe from raw parts fn?
impl<'a, T: Copy> From<(*mut T, usize)> for Buffer<'a, T> {
    fn from(info: (*mut T, usize)) -> Self {
        Buffer {
            ptr: (info.0, null_mut(), 0),
            len: info.1,
            // TODO: use static CPU device or add CPU device to tuple
            device: Device::default(),
            flag: BufFlag::Wrapper,
            p: PhantomData,
        }
    }
}

// TODO: unsafe?
/// A slice is wrapped into a buffer, hence buffer operations can be executed.
/// During these operations, the wrapped slice is updated. (which violates the safety rules / borrow checker of rust)
impl<'a, T> From<&mut [T]> for Buffer<'a, T> {
    fn from(slice: &mut [T]) -> Self {
        Buffer {
            ptr: (slice.as_mut_ptr(), null_mut(), 0),
            len: slice.len(),
            // TODO: use static CPU device or add CPU device to tuple
            device: Device::default(),
            flag: BufFlag::Wrapper,
            p: PhantomData,
        }
    }
}

// TODO: unsafe?
impl<'a, T, const N: usize> From<&mut [T; N]> for Buffer<'a, T> {
    fn from(slice: &mut [T; N]) -> Self {
        Buffer {
            ptr: (slice.as_mut_ptr(), null_mut(), 0),
            len: slice.len(),
            // TODO: use static CPU device or add CPU device to tuple
            device: Device::default(),
            flag: BufFlag::Wrapper,
            p: PhantomData,
        }
    }
}

// TODO: check if Wrapper flag fits
impl<'a, T: CDatatype> From<(*mut c_void, usize)> for Buffer<'a, T> {
    fn from(info: (*mut c_void, usize)) -> Self {
        Buffer {
            ptr: (null_mut(), info.0, 0),
            len: info.1,
            // TODO: use static CPU device or add CPU device to tuple
            device: Device::default(),
            flag: BufFlag::Wrapper,
            p: PhantomData,
        }
    }
}

// TODO: check if Wrapper flag fits
impl<'a, T: CDatatype> From<(u64, usize)> for Buffer<'a, T> {
    fn from(info: (u64, usize)) -> Self {
        Buffer {
            ptr: (null_mut(), null_mut(), info.0),
            len: info.1,
            // TODO: use static CPU device or add CPU device to tuple
            device: Device::default(),
            flag: BufFlag::Wrapper,
            p: PhantomData,
        }
    }
}

/// Adds a buffer to the "cache chain".
/// Following calls will return this buffer,
/// if the corresponding internal count matches with the id used in the cache.
///
///
/// # Example
/// ```
/// use custos::{CPU, AsDev, cached, VecRead, set_count, get_count};
///
/// let device = CPU::new();
/// let dev = device.dev();
/// assert_eq!(0, get_count());
///
/// let mut buf = cached::<f32>(&dev, 10);
/// assert_eq!(1, get_count());
///
/// for value in buf.as_mut_slice() {
///     *value = 1.5;
/// }
///    
/// let new_buf = cached::<i32>(&dev, 10);
/// assert_eq!(2, get_count());
///
/// set_count(0);
/// let buf = cached::<f32>(&dev, 10);
/// assert_eq!(device.read(&buf), vec![1.5; 10]);
/// ```
pub fn cached<T: Default + Copy>(device: &Device, len: usize) -> Buffer<T> {
    let device = get_device!(device, CacheBuf<T>);
    device.cached(len)
}