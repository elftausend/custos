// TODO: move to devices folder ig

use crate::{flag::AllocFlag, prelude::Device, Buffer, HasId, Parents, Shape, StackArray};

pub trait Alloc<T>: Device + Sized {
    /// Allocate memory on the implemented device.
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Alloc, Buffer, Read, flag::AllocFlag, Base, cpu::CPUPtr};
    ///
    /// let device = CPU::<Base>::new();
    /// let data = Alloc::<f32>::alloc::<()>(&device, 12, AllocFlag::None);
    ///
    /// let buf: Buffer = Buffer {
    ///     data,
    ///     device: Some(&device),
    /// };
    /// assert_eq!(vec![0.; 12], device.read(&buf));
    /// ```
    fn alloc<S: Shape>(&self, len: usize, flag: AllocFlag) -> Self::Data<T, S>;

    /// Allocate new memory with data
    /// # Example
    #[cfg_attr(feature = "cpu", doc = "```")]
    #[cfg_attr(not(feature = "cpu"), doc = "```ignore")]
    /// use custos::{CPU, Alloc, Buffer, Read, Base, cpu::CPUPtr};
    ///
    /// let device = CPU::<Base>::new();
    /// let data = Alloc::<i32>::alloc_from_slice::<()>(&device, &[1, 5, 4, 3, 6, 9, 0, 4]);
    ///
    /// let buf: Buffer<i32, CPU> = Buffer {
    ///     data,
    ///     device: Some(&device),
    /// };
    /// assert_eq!(vec![1, 5, 4, 3, 6, 9, 0, 4], device.read(&buf));
    /// ```
    fn alloc_from_slice<S: Shape>(&self, data: &[T]) -> Self::Data<T, S>
    where
        T: Clone;

    /// If the vector `vec` was allocated previously, this function can be used in order to reduce the amount of allocations, which may be faster than using a slice of `vec`.
    #[inline]
    #[cfg(not(feature = "no-std"))]
    fn alloc_from_vec<S: Shape>(&self, vec: Vec<T>) -> Self::Data<T, S>
    where
        T: Clone,
    {
        self.alloc_from_slice(&vec)
    }

    /// Allocates a pointer with the array provided by the `S:`[`Shape`] generic.
    /// By default, the array is flattened and then passed to [`Alloc::alloc_from_slice`].
    #[inline]
    fn alloc_from_array<S: Shape>(&self, array: S::ARR<T>) -> Self::Data<T, S>
    where
        T: Clone,
    {
        let stack_array = StackArray::<S, T>::from_array(array);
        self.alloc_from_slice(stack_array.flatten())
    }
}

pub trait Module<D: Device> {
    type Module;
    // type Data<T, S: Shape>: HasId = D::Data<T, S>;

    fn new() -> Self::Module;
}

pub trait Retriever<T>: Device {
    #[track_caller]
    fn retrieve<S, const NUM_PARENTS: usize>(
        &self,
        len: usize,
        parents: impl Parents<NUM_PARENTS>,
    ) -> Buffer<T, Self, S>
    where
        S: Shape;
}
