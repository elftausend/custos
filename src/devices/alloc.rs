use crate::{flag::AllocFlag, Device, Shape};

use super::StackArray;

pub trait Alloc<T>: Device + Sized {
    /// Allocate memory on the implemented device.
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Alloc, Buffer, Read, flag::AllocFlag, Base, cpu::CPUPtr};
    ///
    /// let device = CPU::<Base>::new();
    /// let data = Alloc::<f32>::alloc::<()>(&device, 12, AllocFlag::None).unwrap();
    ///
    /// let buf: Buffer = Buffer {
    ///     data,
    ///     device: Some(&device),
    /// };
    /// assert_eq!(vec![0.; 12], buf.read());
    /// ```
    fn alloc<S: Shape>(&self, len: usize, flag: AllocFlag) -> crate::Result<Self::Base<T, S>>;

    /// Allocate new memory with data
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Alloc, Buffer, Read, Base, cpu::CPUPtr};
    ///
    /// let device = CPU::<Base>::new();
    /// let data = Alloc::<i32>::alloc_from_slice::<()>(&device, &[1, 5, 4, 3, 6, 9, 0, 4]).unwrap();
    ///
    /// let buf: Buffer<i32, CPU> = Buffer {
    ///     data,
    ///     device: Some(&device),
    /// };
    /// assert_eq!(vec![1, 5, 4, 3, 6, 9, 0, 4], buf.read());
    /// ```
    fn alloc_from_slice<S: Shape>(&self, data: &[T]) -> crate::Result<Self::Base<T, S>>
    where
        T: Clone;

    /// If the vector `vec` was allocated previously, this function can be used in order to reduce the amount of allocations, which may be faster than using a slice of `vec`.
    #[inline]
    #[cfg(feature = "std")]
    fn alloc_from_vec<S: Shape>(&self, vec: Vec<T>) -> crate::Result<Self::Base<T, S>>
    where
        T: Clone,
    {
        self.alloc_from_slice(&vec)
    }

    /// Allocates a pointer with the array provided by the `S:`[`Shape`] generic.
    /// By default, the array is flattened and then passed to [`Alloc::alloc_from_slice`].
    #[inline]
    fn alloc_from_array<S: Shape>(&self, array: S::ARR<T>) -> crate::Result<Self::Base<T, S>>
    where
        T: Clone,
    {
        let stack_array = StackArray::<S, T>::from_array(array);
        self.alloc_from_slice(stack_array.flatten())
    }
}
