use core::{ffi::c_void, mem::ManuallyDrop};

#[cfg(feature = "cpu")]
use crate::cpu::{CPUPtr, CPU};

#[cfg(not(feature = "cpu"))]
use crate::CPU;

use crate::{
    flag::AllocFlag, shape::Shape, Alloc, Base, ClearBuf, CloneBuf, CommonPtrs, Device,
    DevicelessAble, HasId, IsShapeIndep, MainMemory, OnNewBuffer, PtrType, Read, ShallowCopy,
    WriteBuf,
};

pub use self::num::Num;
pub use impl_from_const::*;

mod impl_from;
mod impl_from_const;
mod num;

/// The underlying non-growable array structure of `custos`. A `Buffer` may be encapsulated in other data structures.
/// By default, the `Buffer` is a f32 CPU Buffer with no statically known shape.
/// # Example
#[cfg_attr(feature = "cpu", doc = "```")]
#[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
/// use custos::prelude::*;
///
/// fn buffer_f32_cpu(buf: &Buffer) {}
/// fn buffer_generic<T, D: Device>(buf: &Buffer<T, D>) {}
///
/// let device = CPU::<Base>::new();
/// let buf = Buffer::from((&device, [0.5, 1.3, 3.2, 2.43]));
///
/// buffer_f32_cpu(&buf);
/// buffer_generic(&buf);
/// ```
pub struct Buffer<'a, T = f32, D: Device = CPU<Base>, S: Shape = ()> {
    /// the type of pointer
    pub data: D::Data<T, S>,
    /// A reference to the corresponding device. Mainly used for operations without a device parameter.
    pub device: Option<&'a D>,
}

impl<'a, T, D: Device, S: Shape> Buffer<'a, T, D, S> {
    /// Creates a zeroed (or values set to default) `Buffer` with the given length on the specified device.
    /// This `Buffer` can't outlive the device specified as a parameter.
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Buffer};
    ///
    /// let device = CPU::<Base>::new();
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
    #[inline]
    pub fn new(device: &'a D, len: usize) -> Self
    where
        D: OnNewBuffer<T, D, S> + Alloc<T>,
    {
        let data = device.alloc(len, crate::flag::AllocFlag::None);
        Buffer::from_new_alloc(device, data)
    }

    #[inline]
    fn from_new_alloc(device: &'a D, data: D::Data<T, S>) -> Self
    where
        D: OnNewBuffer<T, D, S>,
    {
        let buf = Buffer {
            data,
            device: Some(device),
        };

        // mind: on_new_buffer must be called for user buffers!
        device.on_new_buffer(device, &buf);
        buf
    }
}

impl<'a, T, D: Device, S: Shape> HasId for Buffer<'a, T, D, S> {
    #[inline]
    fn id(&self) -> super::Id {
        self.data.id()
    }
}

impl<'a, T, D: Device, S: Shape> Drop for Buffer<'a, T, D, S> {
    #[inline]
    fn drop(&mut self) {
        if self.data.flag() != AllocFlag::None {
            return;
        }

        if let Some(device) = self.device {
            device.on_drop_buffer(device, self)
        }
    }
}

impl<'a, T, D: Device + OnNewBuffer<T, D, S>, S: Shape> Buffer<'a, T, D, S> {
    /// Creates a new `Buffer` from a slice (&[T]).
    #[inline]
    pub fn from_slice(device: &'a D, slice: &[T]) -> Self
    where
        T: Clone,
        D: Alloc<T>,
    {
        let data = device.alloc_from_slice(slice);
        Buffer::from_new_alloc(device, data)
    }

    /// Creates a new `Buffer` from a `Vec`.
    #[cfg(not(feature = "no-std"))]
    #[inline]
    pub fn from_vec(device: &'a D, data: Vec<T>) -> Self
    where
        T: Clone,
        D: Alloc<T>,
    {
        let data = device.alloc_from_vec(data);
        Buffer::from_new_alloc(device, data)
    }

    /// Creates a new `Buffer` from an nd-array.
    /// The dimension is defined by the [`Shape`].
    #[inline]
    pub fn from_array(device: &'a D, array: S::ARR<T>) -> Buffer<T, D, S>
    where
        T: Clone,
        D: Alloc<T>,
    {
        let data = device.alloc_from_array(array);
        Buffer::from_new_alloc(device, data)
    }
}

unsafe impl<'a, T, D: Device, S: Shape> Send for Buffer<'a, T, D, S> {}

unsafe impl<'a, T, D: Device, S: Shape> Sync for Buffer<'a, T, D, S> {}

impl<'a, T, D: Device, S: Shape> Buffer<'a, T, D, S> {
    /// Buffers created with this method can outlive the device used to create this `Buffer`.<br>
    /// No operations can be performed on this `Buffer` without a device parameter.
    /// # Examples
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Buffer};
    ///
    /// let mut buf = {
    ///     let device = CPU::<Base>::new();
    ///     Buffer::<u8>::deviceless(&device, 5)
    /// };
    /// // buf.read(); // panics
    /// for (idx, element) in buf.iter_mut().enumerate() {
    ///     *element = idx as u8;
    /// }
    /// assert_eq!(buf.as_slice(), &[0, 1, 2, 3, 4]);
    /// ```
    #[inline]
    pub fn deviceless<'b>(device: &'b D, len: usize) -> Buffer<'a, T, D, S>
    where
        D: DevicelessAble<'b, T, S>,
    {
        Buffer {
            data: device.alloc(len, AllocFlag::None),
            device: None,
        }
    }

    /// Returns the device of the `Buffer`.
    /// Panic if the `Buffer` is deviceless.
    pub fn device(&self) -> &'a D {
        self.device
            .expect("Called device() on a deviceless buffer.")
    }

    /// Reads the contents of the `Buffer`.
    #[inline]
    pub fn read(&'a self) -> D::Read<'a>
    where
        T: Clone + Default,
        D: Read<T, S>,
    {
        self.device().read(self)
    }

    /// Reads the contents of the `Buffer` and writes them into a vector.
    /// `.read` is more efficient, if the device uses host memory.
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Buffer};
    ///
    /// let device = CPU::<Base>::new();
    /// let buf = Buffer::from((&device, [1, 2, 3, 4]));
    ///
    /// assert_eq!(buf.read_to_vec(), vec![1, 2, 3, 4]);
    /// ```
    #[inline]
    #[cfg(not(feature = "no-std"))]
    pub fn read_to_vec(&self) -> Vec<T>
    where
        D: Read<T, S>,
        T: Default + Clone,
    {
        self.device().read_to_vec(self)
    }

    /// Writes a slice to the `Buffer`.
    /// With a CPU buffer, the slice is just copied to the slice of the buffer.
    ///
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Buffer};
    ///
    /// let device = CPU::<Base>::new();
    /// let mut buf = Buffer::<i32>::new(&device, 6);
    /// buf.write(&[4, 2, 3, 4, 5, 3]);
    ///
    /// assert_eq!(&*buf, [4, 2, 3, 4, 5, 3]);
    /// ```
    #[inline]
    pub fn write(&mut self, data: &[T])
    where
        D: WriteBuf<T, S, D>,
    {
        self.device().write(self, data)
    }

    /// Writes the contents of the source buffer to self.
    #[inline]
    pub fn write_buf(&mut self, src: &Buffer<T, D, S>)
    where
        T: Clone,
        D: WriteBuf<T, S, D>,
    {
        self.device().write_buf(self, src)
    }

    /// Returns the number of elements contained in `Buffer`.
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Buffer};
    ///
    /// let device = CPU::<Base>::new();
    /// let a = Buffer::<i32, _>::new(&device, 10);
    /// assert_eq!(a.len(), 10)
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.data.size()
    }

    /// Creates a shallow copy of &self.
    ///
    /// # Safety
    /// Itself, this function does not need to be unsafe.
    /// However, declaring this function as unsafe highlights the violation of creating two or more owners for one resource.
    /// Furthermore, the resulting `Buffer` can outlive `self`.
    #[inline]
    pub unsafe fn shallow(&self) -> Buffer<'a, T, D, S>
    where
        <D as Device>::Data<T, S>: ShallowCopy,
    {
        Buffer {
            data: self.data.shallow(),
            device: self.device,
        }
    }

    /// Returns a shallow copy of &self, if the `realloc` feature is deactivated.
    /// If the `realloc` feature is activated, it returns a deep copy / clone.
    ///
    /// # Safety
    /// Itself, this function does not need to be unsafe.
    /// However, declaring this function as unsafe highlights the violation of possibly creating two or more owners for one resource.
    /// Furthermore, the resulting `Buffer` can outlive `self`.
    pub unsafe fn shallow_or_clone(&self) -> Buffer<'a, T, D, S>
    where
        <D as Device>::Data<T, S>: ShallowCopy,
        T: Clone,
        D: CloneBuf<'a, T, S>,
    {
        {
            #[cfg(not(feature = "realloc"))]
            self.shallow()
        }

        #[cfg(feature = "realloc")]
        self.clone()
    }

    /// Sets all elements in `Buffer` to the default value.
    pub fn clear(&mut self)
    where
        D: ClearBuf<T, S, D>,
    {
        self.device().clear(self)
    }
}

// TODO better solution for the to_dims stack problem?
impl<'a, T, D: Device, S: Shape> Buffer<'a, T, D, S> {
    /// Converts a non stack allocated `Buffer` with shape `S` to a `Buffer` with shape `O`.
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Buffer, Shape, Dim1, Dim2};
    ///
    /// let device = CPU::<Base>::new();
    /// let a = Buffer::<i32, CPU, Dim1<10>>::new(&device, 10);
    /// let _b = a.to_dims::<Dim2<5, 2>>();
    ///
    /// ```
    #[inline]
    pub fn to_dims<O: Shape>(self) -> Buffer<'a, T, D, O>
    where
        D: crate::ToDim<T, S, O>,
        D::Data<T, S>: ShallowCopy,
    {
        let buf = ManuallyDrop::new(self);

        let data = buf.device().to_dim(unsafe { buf.data.shallow() });

        Buffer {
            data,
            device: buf.device,
        }
    }
}

impl<'a, T, D: IsShapeIndep, S: Shape> Buffer<'a, T, D, S> {
    /// Returns a reference of the same buffer, but with a different shape.
    /// The Buffer is shape independet, so it can be converted to any shape.
    #[inline]
    pub fn as_dims<'b, O: Shape>(&self) -> &Buffer<'b, T, D, O> {
        // Safety: shape independent buffers
        // -> all dims have a size of 0
        // -> all other buffer types do not depend on any features of the shape (S::ARR).
        unsafe { &*(self as *const Self).cast() }
    }

    /// Returns a mutable reference of the same buffer, but with a different shape.
    /// The Buffer is shape independet, so it can be converted to any shape.
    #[inline]
    pub fn as_dims_mut<'b, O: Shape>(&mut self) -> &mut Buffer<'b, T, D, O> {
        unsafe { &mut *(self as *mut Self).cast() }
    }
}

impl<'a, T, D: Device, S: Shape> Buffer<'a, T, D, S>
where
    D::Data<T, S>: CommonPtrs<T>,
{
    #[inline]
    /// Returns all types of pointers. (host, OpenCL, CUDA)
    pub fn ptrs(&self) -> (*const T, *mut c_void, u64) {
        self.data.ptrs()
    }

    #[inline]
    /// Returns all types of pointers. (host, OpenCL, CUDA)
    pub fn ptrs_mut(&mut self) -> (*mut T, *mut c_void, u64) {
        self.data.ptrs_mut()
    }
}

impl<'a, T, D: Device> Buffer<'a, T, D> {
    /// Returns `true` if `Buffer` is created without a slice.
    /// # Example
    /// ```
    /// use custos::{CPU, Buffer};
    ///
    /// let a = Buffer::<i32, ()>::from(5);
    /// assert!(a.is_empty())
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(feature = "cpu")]
impl<'a, T, S: Shape> Buffer<'a, T, CPU, S> {
    /// Constructs a deviceless `Buffer` out of a host pointer and a length.
    /// # Example
    /// ```
    /// use custos::{Buffer, Alloc, CPU, Read, flag::AllocFlag};
    /// use std::ffi::c_void;
    ///
    /// let device = CPU::<Base>::new();
    /// let mut ptr = Alloc::<f32>::alloc(&device, 10, AllocFlag::None);
    /// let mut buf = unsafe {
    ///     Buffer::<_, _, ()>::from_raw_host(ptr.ptr, 10)
    /// };
    /// for (idx, value) in buf.iter_mut().enumerate() {
    ///     *value += idx as f32;
    /// }
    ///
    /// assert_eq!(buf.as_slice(), &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,]);
    ///
    /// ```
    /// # Safety
    /// The pointer must be valid.
    /// The `Buffer` does not manage deallocation of the allocated memory.
    #[inline]
    pub unsafe fn from_raw_host(ptr: *mut T, len: usize) -> Buffer<'a, T, CPU, S> {
        Buffer {
            data: CPUPtr::from_ptr(ptr, len, AllocFlag::Wrapper),
            device: None,
        }
    }

    /// Constructs a `Buffer` out of a host pointer and a length.
    /// The provided device can be used to shorten operation calls.
    ///
    /// # Safety
    /// The pointer must be valid.
    /// The `Buffer` does not manage deallocation of the allocated memory.
    #[inline]
    pub unsafe fn from_raw_host_device(
        device: &'a CPU,
        ptr: *mut T,
        len: usize,
    ) -> Buffer<'a, T, CPU, S> {
        Buffer {
            data: CPUPtr::from_ptr(ptr, len, AllocFlag::Wrapper),
            device: Some(device),
        }
    }
}

#[cfg(feature = "opencl")]
impl<'a, T, S: Shape> Buffer<'a, T, crate::OpenCL, S> {
    /// Returns the OpenCL pointer of the `Buffer`.
    #[inline]
    pub fn cl_ptr(&self) -> *mut c_void {
        assert!(
            !self.data.ptr.is_null(),
            "called cl_ptr() on an invalid OpenCL buffer"
        );
        self.ptrs().1
    }
}

#[cfg(feature = "cuda")]
impl<'a, T> Buffer<'a, T, crate::CUDA> {
    // TODO: replace buf.data.2 with this fn, do the same with cl, cpu
    /// Returns a non null CUDA pointer
    #[inline]
    pub fn cu_ptr(&self) -> u64 {
        assert!(
            self.ptrs().2 != 0,
            "called cu_ptr() on an invalid CUDA buffer"
        );
        self.data.ptr
    }
}

impl<'a, T, D: MainMemory, S: Shape> Buffer<'a, T, D, S> {
    /// Returns a CPU slice. This does not work with CUDA or raw OpenCL buffers.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self
    }

    /// Returns a mutable CPU slice.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

// custos v0.5 compatability
impl<'a, T, D: MainMemory, S: Shape> Buffer<'a, T, D, S>
where
    D::Data<T, S>: CommonPtrs<T>,
{
    /// Returns a non null host pointer
    #[inline]
    pub fn host_ptr(&self) -> *const T {
        assert!(
            !self.ptrs().0.is_null(),
            "called host_ptr() on an invalid CPU buffer (this would dereference a null pointer)"
        );
        self.ptrs().0
    }

    /// Returns a non null host pointer
    #[inline]
    pub fn host_ptr_mut(&mut self) -> *mut T {
        assert!(
            !self.ptrs().0.is_null(),
            "called host_ptr_mut() on an invalid CPU buffer (this would dereference a null pointer)"
        );
        self.ptrs_mut().0
    }
}

impl<'a, T, D, S> Clone for Buffer<'a, T, D, S>
where
    T: Clone,
    D: CloneBuf<'a, T, S> + Device,
    S: Shape,
{
    fn clone(&self) -> Self {
        self.device().clone_buf(self)
    }
}

/*#[cfg(feature = "safe")]
unsafe impl<T> Send for Buffer<'a, T> {}
#[cfg(feature = "safe")]
unsafe impl<T> Sync for Buffer<'a, T> {}*/

impl<'a, T, D: Device, S: Shape> Default for Buffer<'a, T, D, S>
where
    D::Data<T, S>: Default,
{
    fn default() -> Self {
        Self {
            data: D::Data::<T, S>::default(),
            device: None,
        }
    }
}

impl<T, D: MainMemory> AsRef<[T]> for Buffer<'_, T, D> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self
    }
}

impl<T, D: MainMemory> AsMut<[T]> for Buffer<'_, T, D> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}

/// A main memory `Buffer` dereferences into a slice.
///
/// # Examples
///
#[cfg_attr(feature = "cpu", doc = "```")]
#[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
/// use custos::{Buffer, CPU};
///
/// let device = CPU::<Base>::new();
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
impl<T, D: MainMemory, S: Shape> core::ops::Deref for Buffer<'_, T, D, S> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { core::slice::from_raw_parts(D::as_ptr(&self.data), self.len()) }
    }
}

/// A main memory `Buffer` dereferences into a mutable slice.
///
/// # Examples
///
#[cfg_attr(feature = "cpu", doc = "```")]
#[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
/// use custos::{Buffer, CPU};
///  
/// let device = CPU::<Base>::new();
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
impl<T, D: MainMemory, S: Shape> core::ops::DerefMut for Buffer<'_, T, D, S> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { core::slice::from_raw_parts_mut(D::as_ptr_mut(&mut self.data), self.len()) }
    }
}

#[cfg(not(feature = "no-std"))]
use core::fmt::Debug;

#[cfg(not(feature = "no-std"))]
impl<'a, T, D> Debug for Buffer<'a, T, D>
where
    T: Debug + Default + Clone + 'a,
    D: Read<T> + Device + 'a,
    for<'b> <D as Read<T>>::Read<'b>: Debug,
    D::Data<T, ()>: CommonPtrs<T>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Buffer")
            .field("ptr (CPU, CL, CU)", &self.ptrs())
            .field("len", &self.len());
        writeln!(f, ",")?;

        if !self.ptrs().0.is_null() {
            let slice = unsafe { std::slice::from_raw_parts(self.ptrs().0, self.len()) };
            writeln!(f, "CPU:    {slice:?}")?;
        }

        #[cfg(feature = "opencl")]
        if !self.ptrs().1.is_null() {
            write!(f, "OpenCL: {:?}, ", self.read())?;
        }

        #[cfg(feature = "cuda")]
        if self.ptrs().2 != 0 {
            write!(f, "CUDA: {:?}, ", self.read())?;
        }

        write!(
            f,
            "datatype={}, device={device} }}",
            core::any::type_name::<T>(),
            device = core::any::type_name::<D>()
        )
    }
}

impl<'a, T, D: MainMemory, S: Shape> core::iter::IntoIterator for &'a Buffer<'_, T, D, S> {
    type Item = &'a T;

    type IntoIter = core::slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, D: MainMemory, S: Shape> core::iter::IntoIterator for &'a mut Buffer<'_, T, D, S> {
    type Item = &'a mut T;

    type IntoIter = core::slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use crate::Buffer;

    #[cfg(feature = "cpu")]
    #[test]
    fn test_deref() {
        use crate::Base;

        let device = crate::CPU::<Base>::new();
        let buf: Buffer<i32> = Buffer::from((&device, [1, 2, 3, 4]));
        let slice = &*buf;
        assert_eq!(slice, &[1, 2, 3, 4]);
    }

    #[cfg(feature = "opencl")]
    #[cfg(unified_cl)]
    #[test]
    fn test_deref_cl() -> crate::Result<()> {
        use crate::OpenCL;

        let device = OpenCL::new(0)?;
        let buf = Buffer::from((&device, [1, 2, 3, 4]));
        let slice = &*buf;
        assert_eq!(slice, &[1, 2, 3, 4]);

        Ok(())
    }

    #[cfg(feature = "stack")]
    #[test]
    fn test_deref_stack() -> crate::Result<()> {
        use crate::{shape::Dim1, stack::Stack};

        let dev = Stack::new();

        //TODO
        let buf = Buffer::<i32, _, Dim1<4>>::from((&dev, [1i32, 2, 3, 4]));
        let slice = &*buf;
        assert_eq!(slice, &[1, 2, 3, 4]);

        Ok(())
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_debug_print() {
        use crate::Base;

        let device = crate::CPU::<Base>::new();
        let buf = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

        println!("{buf:?}",);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_to_dims() {
        use crate::{Base, Dim2};

        let device = crate::CPU::<Base>::new();
        let buf = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));
        let buf_dim2 = buf.to_dims::<Dim2<3, 2>>();

        buf_dim2.to_dims::<()>();
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_id_cpu() {
        use crate::{Base, HasId, CPU};

        let device = CPU::<Base>::new();

        let buf = Buffer::from((&device, [1, 2, 3, 4]));
        assert_eq!(buf.id(), buf.data.id())
    }

    #[cfg(feature = "stack")]
    #[cfg(not(feature = "no-std"))]
    #[should_panic]
    #[test]
    fn test_id_stack() {
        use crate::{Stack, WithShape, HasId};

        let device = Stack::new();

        let buf = Buffer::with(&device, [1, 2, 3, 4]);
        buf.id();
    }
}
