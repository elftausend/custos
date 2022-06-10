use std::{ffi::c_void, ptr::null_mut, fmt::Debug};

#[cfg(feature="opencl")]
#[cfg(feature="safe")]
use crate::opencl::api::release_mem_object;
use crate::{Device, GenericOCL, get_device, CacheBuf, opencl::api::retain_mem_object, VecRead};

#[cfg(not(feature="safe"))]
use crate::number::Number;


#[cfg_attr(not(feature = "safe"), derive(Debug, Clone, Copy))]
pub struct Buffer<T> {
    pub ptr: (*mut T, *mut c_void),
    pub len: usize,
}

impl<T> Buffer<T> {
    /// Creates an empty buffer with the given length on the specified device.
    /// ```
    /// use custos::{CPU, Buffer};
    /// 
    /// let device = CPU::new();
    /// let buffer = Buffer::<i32>::new(&device, 6);
    /// 
    /// // this works only with cpu buffers
    /// let slice = buffer.as_slice();
    /// 
    /// assert_eq!(slice, &[0; 6]);
    /// 
    /// ```
    pub fn new<D: Device<T>>(device: &D, len: usize) -> Buffer<T> {
        Buffer {
            ptr: device.alloc(len),
            len,
        }
    }

    /// Returns a CPU slice.
    pub fn as_slice(&self) -> &[T] {
        assert!(self.ptr.0 != std::ptr::null_mut(), "called as_slice() on a non CPU buffer (this would dereference a null pointer)");
        unsafe {
            std::slice::from_raw_parts(self.ptr.0, self.len)
        }
    }
    
    /// Returns a mutable CPU slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        assert!(self.ptr.0 != std::ptr::null_mut(), "called as_mut_slice() on a non CPU buffer (this would dereference a null pointer)");
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr.0, self.len)
        }
    }

    #[cfg(not(feature="safe"))]
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
    /// let x: Buffer<f32> = (&mut [5., 4., 8.]).into();
    /// assert_eq!(x.item(), 0.);
    /// ```
    pub fn item(&self) -> T {
        if self.len == 0 {
            return unsafe { *self.ptr.0 };
        }
        T::default()
    }
}

#[cfg(feature="safe")]
unsafe impl<T> Send for Buffer<T> {}
#[cfg(feature="safe")]
unsafe impl<T> Sync for Buffer<T> {}


#[cfg(feature="safe")]
impl<T> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        #[cfg(feature="opencl")]
        if !self.ptr.1.is_null() { 
            retain_mem_object(self.ptr.1).unwrap();
        };
        Self { ptr: self.ptr, len: self.len}
    }
}

#[cfg(feature="safe")]
impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.0.is_null() && self.ptr.1.is_null() {
                Box::from_raw(self.ptr.0);
            }
            
            #[cfg(feature="opencl")]
            if !self.ptr.1.is_null() {
                release_mem_object(self.ptr.1).unwrap()
            }
        }
    }
}

impl<T> Default for Buffer<T> {
    fn default() -> Self {
        Self { ptr: (null_mut(), null_mut()), len: Default::default() }
    }
}

impl<T> AsRef<[T]> for Buffer<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> std::ops::Deref for Buffer<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> std::ops::DerefMut for Buffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T: Debug + Default + Copy> Debug for Buffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer").field("ptr (CPU, OpenCL)", &self.ptr).field("len", &self.len);
        writeln!(f, ",")?;
        if self.ptr.0 != null_mut() {
            writeln!(f, "CPU:    {:?}", self.as_slice())?; 
        }

        if self.ptr.1 != null_mut() {
            let read = get_device!(VecRead, T).unwrap();
            write!(f, "OpenCL: {:?}, ", read.read(self))?; 
        }

        write!(f, "datatype={} }}", std::any::type_name::<T>())
    }
}

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


#[cfg(not(feature="safe"))]
impl<T: Number> From<T> for Buffer<T> {
    fn from(val: T) -> Self {
        Buffer { 
            ptr: ( Box::into_raw(Box::new(val)), null_mut() ), 
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

#[cfg(not(feature="safe"))]
impl<T: Copy> From<(*mut T, usize)> for Buffer<T> {
    fn from(info: (*mut T, usize)) -> Self {
        Buffer {
            ptr: (info.0, null_mut()),
            len: info.1,
        } 
    }
}

#[cfg(not(feature="safe"))]
impl<const N: usize, T> From<&mut [T; N]> for Buffer<T> where T: Copy {
    fn from(arr: &mut [T; N]) -> Self {
        Buffer { 
            ptr: (arr.as_mut_ptr(), null_mut()), 
            len: N
        }
    }
}


impl<T: GenericOCL> From<(*mut c_void, usize)> for Buffer<T> {
    fn from(info: (*mut c_void, usize)) -> Self {
        Buffer {
            ptr: (null_mut(), info.0),
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
/// An empty buffer with the specified len is returned.
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
pub fn cached<T: GenericOCL>(len: usize) -> Buffer<T> {
    let device = get_device!(CacheBuf, T).unwrap();
    device.cached_buf(len)
}