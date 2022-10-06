use crate::{cpu::CPUPtr, Alloc, BufFlag, Buffer, Device, GraphReturn, PtrType, VecRead, Node};

use super::{static_cpu, StaticGPU};


impl<'a, T: Clone> Buffer<'a, T> {
    /// Moves the buffer [`Buffer`] to a static device.<br>
    /// This device is chosen via the type parameter D -> [`OpenCL`], [`CUDA`].<br>
    /// It is recommended to use the to_gpu() method of [`Buffer`].
    ///  
    /// Example
    #[cfg_attr(not(feature = "opencl"), doc = "```ignore")]
    #[cfg_attr(feature = "opencl", doc = "```")]
    /// use custos::prelude::*;
    /// 
    /// let cpu_buffer = Buffer::from(&[1., 2., 3.]);
    ///
    /// let cl_buf = cpu_buffer.to_dev::<OpenCL>();
    /// assert_eq!(cl_buf.read(), vec![1., 2., 3.]);
    /// ```
    #[inline]
    pub fn to_dev<D>(self) -> Buffer<'static, T, D> 
    where
        D: Device + StaticGPU + Alloc + GraphReturn 
    {

        Buffer::from((D::as_static(), self.as_slice()))
    }

    /// Moves a [`Buffer`] to a [`CUDA`] device.<br>
    /// It is recommended to use the to_gpu() method of [Buffer].
    ///  
    /// Example
    /// ```
    /// use custos::prelude::*;
    /// 
    /// let cpu_buffer = Buffer::from(&[1., 2., 3.]);
    ///
    /// let cu_buf = cpu_buffer.to_cuda();
    /// assert_eq!(cu_buf.read(), vec![1., 2., 3.]);
    /// ```
    #[cfg(feature = "cuda")]
    #[inline]
    pub fn to_cuda(self) -> Buffer<'a, T, crate::CUDA> {
        self.to_dev::<crate::CUDA>()
    }

    /// Converts a [Buffer] to an OpenCL device buffer.<br>
    /// It is recommended to use the to_gpu() method of [Buffer].
    ///  
    /// Example
    /// ```
    /// use custos::prelude::*;
    /// 
    /// let cpu_buffer = Buffer::from(&[1., 2., 3.]);
    ///
    /// let cl_buf = cpu_buffer.to_cl();
    /// assert_eq!(cl_buf.read(), vec![1., 2., 3.]);
    /// ```
    #[cfg(feature = "opencl")]
    #[inline]
    pub fn to_cl(self) -> Buffer<'a, T, crate::OpenCL> {
        self.to_dev::<crate::OpenCL>()
    }

    /// Converts a [Buffer] to an OpenCL device buffer.
    /// 
    /// This method depends on the feature configuration.<br>
    /// If the 'cuda' feature is enabled, this function will return a CUDA device buffer.
    /// 
    /// # Example
    /// ```
    /// use custos::prelude::*;
    /// 
    /// let cpu_buf = Buffer::from(&[4., 3., 5.]);
    /// let cl_buf = cpu_buf.to_gpu();
    /// 
    /// assert_eq!(cl_buf.read(), vec![4., 3., 5.]);
    /// ```
    #[cfg(feature = "opencl")]
    #[cfg(not(feature = "cuda"))]
    #[inline]
    pub fn to_gpu(self) -> Buffer<'a, T, crate::OpenCL> {
        self.to_cl()
    }

    /// Converts a [Buffer] to a CUDA device buffer.
    /// 
    /// This method depends on the feature configuration.<br>
    /// If the 'cuda' feature is disabled, this function will return an OpenCL device buffer.
    /// 
    /// # Example
    /// ```
    /// use custos::prelude::*;
    /// 
    /// let cpu_buf = Buffer::from(&[4., 3., 5.]);
    /// let cuda_buf = cpu_buf.to_gpu();
    /// 
    /// assert_eq!(cuda_buf.read(), vec![4., 3., 5.]);
    /// ```
    #[cfg(feature = "cuda")]
    #[inline]
    pub fn to_gpu(self) -> Buffer<'a, T, crate::CUDA> {
        self.to_cuda()
    }
}

impl<'a, T, D> Buffer<'a, T, D> 
where 
    T: Clone + Default, 
    D: Device + VecRead<T, D> 
{
    /// Moves the [`Buffer`] back to a CPU buffer.
    /// 
    /// Example
    /// 
    #[cfg_attr(not(any(feature = "opencl", feature="cuda")), doc = "```ignore")]
    #[cfg_attr(any(feature = "opencl", feature="cuda"), doc = "```")]
    /// use custos::prelude::*;
    /// 
    /// let gpu_buf = Buffer::from(&[1, 2, 3]).to_gpu();
    /// 
    /// // ... some operations ...
    /// 
    /// let cpu_buf = gpu_buf.to_cpu();
    /// assert_eq!(cpu_buf.as_slice(), &[1, 2, 3]);
    /// ```
    #[inline]
    pub fn to_cpu(self) -> Buffer<'a, T> {
        Buffer::from((static_cpu(), self.read()))
    }
}

impl<'a, T: Clone> From<&[T]> for Buffer<'a, T> {
    fn from(slice: &[T]) -> Self {
        let device = static_cpu();
        Buffer {
            ptr: CPUPtr::from_ptrs(device.with_data(slice)),
            len: slice.len(),
            device: Some(device),
            flag: BufFlag::None,
            node: device.graph().add_leaf(slice.len()),
        }
    }
}

impl<'a, T: Clone, const N: usize> From<&[T; N]> for Buffer<'a, T> {
    fn from(slice: &[T; N]) -> Self {
        let device = static_cpu();
        Buffer {
            ptr: CPUPtr::from_ptrs(device.with_data(slice)),
            len: slice.len(),
            device: Some(device),
            flag: BufFlag::None,
            node: device.graph().add_leaf(slice.len()),
        }
    }
}

impl<'a, T: Clone, const N: usize> From<[T; N]> for Buffer<'a, T> {
    fn from(slice: [T; N]) -> Self {
        let device = static_cpu();
        Buffer {
            ptr: CPUPtr::from_ptrs(device.with_data(&slice)),
            len: slice.len(),
            device: Some(device),
            flag: BufFlag::None,
            node: Node::default(),
        }
    }
}

impl<'a, T: Clone> From<Vec<T>> for Buffer<'a, T> {
    fn from(data: Vec<T>) -> Self {
        let device = static_cpu();
        Buffer {
            len: data.len(),
            ptr: CPUPtr::from_ptrs(device.alloc_with_vec(data)),
            device: Some(device),
            flag: BufFlag::None,
            node: Node::default(),
        }
    }
}
