use std::alloc::Layout;
use std::iter::FromIterator;
use std::{ffi::c_void, fmt::Debug, ptr::null_mut};

#[cfg(feature = "opencl")]
use crate::opencl::api::release_mem_object;
use crate::{
    Alloc, CDatatype, ClearBuf, CloneBuf, GraphReturn, Node,
    VecRead, WriteBuf, GLOBAL_CPU, Deviceless, DevicelessAble, CPU, CacheBuf,
};

/// Descripes the type of a [`Buffer`]
#[derive(Debug, Clone, Copy)]
pub enum BufFlag {
    None,
    Cache,
    Wrapper,
    Item,
}

impl Default for BufFlag {
    fn default() -> Self {
        BufFlag::None
    }
}

impl PartialEq for BufFlag {
    fn eq(&self, other: &Self) -> bool {
        core::mem::discriminant(self) == core::mem::discriminant(other)
    }
}

/// The underlying non-growable array structure. A `Buffer` may be encapsulated in other structs.
pub struct Buffer<'a, T, D> {
    pub ptr: (*mut T, *mut c_void, u64),
    pub len: usize,
    pub device: Option<&'a D>,
    pub flag: BufFlag,
    pub node: Node,
}

impl<'a, T> Buffer<'a, T, Deviceless> 
where 
    
{
    /// Creates a zeroed (or values set to default) `Buffer` with the given length on the specified device.
    /// This `Buffer` can't outlive the device specified as a parameter.
    /// ```
    /// use custos::{CPU, Buffer};
    ///
    /// let device = CPU::new();
    /// let mut buffer = Buffer::<i32, _>::new(&device, 6);
    ///
    /// // this works only with cpu buffers (this creates a slice with the host pointer)
    /// for value in &mut buffer {
    ///     *value = 2;
    /// }
    ///
    /// assert_eq!(buffer.as_slice(), &[2; 6]);
    ///
    /// ```
    pub fn new<D>(device: &'a D, len: usize) -> Buffer<'a, T, D>
    where
        D: Alloc + GraphReturn,
    {
        Buffer {
            ptr: device.alloc(len),
            len,
            device: Some(device),
            node: device.graph().add_leaf(len),
            ..Default::default()
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
    ///     Buffer::<u8, _>::deviceless(&device, 5)
    /// };
    /// // buf.read(); // panics
    /// for (idx, element) in buf.iter_mut().enumerate() {
    ///     *element = idx as u8;
    /// }
    /// assert_eq!(buf.as_slice(), &[0, 1, 2, 3, 4]);
    /// ```
    pub fn deviceless<'b>(device: &'b impl DevicelessAble, len: usize) -> Buffer<'a, T, Deviceless> {
        Buffer {
            ptr: device.alloc(len),
            len,
            ..Default::default()
        }
    }

}

impl<'a, T, D> Buffer<'a, T, D> {
    pub fn device(&self) -> &'a D {
        self.device.expect("Called device() on a deviceless buffer.")
    }
    /// Constructs a `Buffer` out of a host pointer and a length.
    /// # Example
    /// ```
    /// use custos::{Buffer, Alloc, CPU, VecRead};
    /// use std::ffi::c_void;
    ///
    /// let device = CPU::new();
    /// let ptrs: (*mut f32, *mut c_void, u64) = device.alloc(10);
    /// let mut buf = unsafe {
    ///     Buffer::<f32, CPU>::from_raw_host(ptrs.0, 10)
    /// };
    /// for (idx, value) in buf.iter_mut().enumerate() {
    ///     *value += idx as f32;
    /// }
    /// assert_eq!(buf.as_slice(), &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,])
    ///
    /// ```
    /// # Safety
    /// The pointer must not outlive the Buffer.
    pub unsafe fn from_raw_host(ptr: *mut T, len: usize) -> Buffer<'a, T, Deviceless> {
        Buffer {
            ptr: (ptr, null_mut(), 0),
            len,
            ..Default::default()
        }
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
    /// use custos::{CPU, Buffer, Deviceless};
    ///
    /// let a = Buffer::<i32, Deviceless>::from(5);
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
        assert!(self.ptr.2 != 0, "called cu_ptr() on an invalid CUDA buffer");
        self.ptr.2
    }

    /// Returns a CPU slice. This does not work with CUDA or OpenCL buffers.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        assert!(
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
        if self.len == 0 && !self.ptr.0.is_null() {
            return unsafe { *self.ptr.0 };
        }
        T::default()
    }

    /// Sets all elements in `Buffer` to the default value.
    pub fn clear(&mut self)
    where
        T: CDatatype,
        D: ClearBuf<T>,
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
        D: VecRead<T>
    {
        self.device().read(self)
    }

    /// Writes a slice to the Buffer.
    /// With a CPU buffer, the slice is just copied to the slice of the buffer.
    pub fn write(&mut self, data: &[T])
    where
        T: Copy,
        D: WriteBuf<T>
    {
        self.device().write(self, data)
    }

    /*#[cfg(feature = "cuda")]
    pub fn to_cuda<'c>(&self, cuda_device: &'c crate::CudaDevice) -> crate::Result<Buffer<'c, T>> {
        use crate::{DeviceError, DeviceType};

        if self.device.device_type != DeviceType::CPU {
            return Err(DeviceError::CPUtoCUDA.into());
        }

        let mut out = crate::Cache::get(cuda_device, self.len, );
        cuda_device.write(&mut out, self);
        Ok(out)
    }*/

    /// Creates a shallow copy of &self.
    ///
    /// # Safety
    /// Itself, this function does not need to be unsafe.
    /// However, declaring this function as unsafe highlights the violation of creating two or more owners for one resource.
    /// Furthermore, the resulting `Buffer` can outlive `self`.
    pub unsafe fn shallow(&self) -> Buffer<'a, T, D> {
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
        T: Clone,
        D: CloneBuf<'a, T>
    {
        {
            #[cfg(not(feature = "realloc"))]
            self.shallow()
        }

        #[cfg(feature = "realloc")]
        self.clone()
    }
}

impl<'a, T: Clone, D: CloneBuf<'a, T>> Clone for Buffer<'a, T, D> {
    fn clone(&self) -> Self {
        //get_device!(self.device, CloneBuf<T>).clone_buf(self)
        self.device().clone_buf(self)
    }
}

/*#[cfg(feature = "safe")]
unsafe impl<T> Send for Buffer<'a, T> {}
#[cfg(feature = "safe")]
unsafe impl<T> Sync for Buffer<'a, T> {}*/

impl<'a, A: Clone + Default> FromIterator<A> for Buffer<'a, A, CPU> {
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        // Safety: GLOBAL_CPU should live long enough
        let device = unsafe {
            GLOBAL_CPU.with(|device| {
                device as *const CPU
            }).as_ref().unwrap()
        };
        let from_iter = Vec::from_iter(iter);

        Buffer {
            len: from_iter.len(),
            node: device.graph().add_leaf(from_iter.len()),
            ptr: device.alloc_with_vec(from_iter),
            device: Some(device),
            flag: BufFlag::None
        }
    }
}

impl<T, D> Drop for Buffer<'_, T, D> {
    fn drop(&mut self) {
        if self.flag == BufFlag::Item {
            drop(unsafe { Box::from_raw(self.ptr.0) });
            return;
        }
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

impl<'a, T, D> Default for Buffer<'a, T, D> {
    fn default() -> Self {
        Self {
            ptr: (null_mut(), null_mut(), 0),
            flag: BufFlag::default(),
            len: Default::default(),
            device: None,
            node: Node::default(),
        }
    }
}

impl<T> AsRef<[T]> for Buffer<'_, T, CPU> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for Buffer<'_, T, CPU> {
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
impl<T, D> std::ops::Deref for Buffer<'_, T, D> {
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
impl<T, D> std::ops::DerefMut for Buffer<'_, T, D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T: Debug + Default + Copy, D: VecRead<T>> Debug for Buffer<'_, T, D> {
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
            write!(f, "OpenCL: {:?}, ", self.device().read(self))?;
        }

        #[cfg(feature = "cuda")]
        if self.ptr.2 != 0 {
            write!(f, "CUDA: {:?}, ", self.device().read(self))?;
        }

        write!(
            f,
            "datatype={}, device={device:?} }}",
            std::any::type_name::<T>(),
            device = std::any::type_name::<D>()
        )
    }
}

impl<'a, T, D> std::iter::IntoIterator for &'a Buffer<'_, T, D> {
    type Item = &'a T;

    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, D> std::iter::IntoIterator for &'a mut Buffer<'_, T, D> {
    type Item = &'a mut T;

    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T: crate::number::Number> From<T> for Buffer<'_, T, Deviceless> {
    fn from(val: T) -> Self {
        Buffer {
            ptr: (Box::into_raw(Box::new(val)), null_mut(), 0),
            len: 0,
            flag: BufFlag::Item,
            ..Default::default()
        }
    }
}

impl<'a, T: Clone, D: Alloc + GraphReturn, const N: usize> From<(&'a D, [T; N])>
    for Buffer<'a, T, D>
{
    fn from(device_slice: (&'a D, [T; N])) -> Self {
        let len = device_slice.1.len();
        Buffer {
            ptr: device_slice.0.with_data(&device_slice.1),
            len,
            device: Some(device_slice.0),
            node: device_slice.0.graph().add_leaf(len),
            ..Default::default()
        }
    }
}

impl<'a, T: Clone, D: Alloc + GraphReturn> From<(&'a D, &[T])> for Buffer<'a, T, D> {
    fn from(device_slice: (&'a D, &[T])) -> Self {
        let len = device_slice.1.len();
        Buffer {
            ptr: device_slice.0.with_data(device_slice.1),
            len,
            device: Some(device_slice.0),
            node: device_slice.0.graph().add_leaf(len),
            ..Default::default()
        }
    }
}

impl<'a, T: Clone, D: Alloc + GraphReturn> From<(&'a D, Vec<T>)> for Buffer<'a, T, D> {
    fn from(device_slice: (&'a D, Vec<T>)) -> Self {
        let len = device_slice.1.len();
        Buffer {
            ptr: device_slice.0.alloc_with_vec(device_slice.1),
            len,
            device: Some(device_slice.0),
            node: device_slice.0.graph().add_leaf(len),
            ..Default::default()
        }
    }
}

impl<'a, T: Clone, D: Alloc + GraphReturn> From<(&'a D, &Vec<T>)> for Buffer<'a, T, D> {
    fn from(device_slice: (&'a D, &Vec<T>)) -> Self {
        let len = device_slice.1.len();
        Buffer {
            ptr: device_slice.0.with_data(device_slice.1),
            len,
            device: Some(device_slice.0),
            node: device_slice.0.graph().add_leaf(len),
            ..Default::default()
        }
    }
}

// TODO: Think of adding them to the graph
/*
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
*/

#[cfg_attr(feature = "realloc", doc = "```ignore")]
/// Adds a `Buffer` to the "cache chain".
/// Following calls will return this `Buffer`,
/// if the corresponding internal count matches with the id used in the cache.
///
///
/// # Example
/// ```
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
pub fn cached<'a, T, D: CacheBuf<'a, T>>(device: &'a D, len: usize) -> Buffer<'a, T, D> {
    device.cached(len)
}