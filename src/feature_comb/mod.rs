mod features;
pub use features::*;

mod modules;
pub use modules::*;

use crate::{cpu::CPUPtr, flag::AllocFlag, Shape, StackArray};

struct CPU<Mods> {
    modules: Mods,
}

pub trait Device {}

impl<Mods> Device for CPU<Mods> {}

pub trait Alloc {
    type Data<T, S: Shape>;

    fn alloc<T, S: Shape>(&self, len: usize, flag: AllocFlag) -> Self::Data<T, S>;
    fn alloc_from_slice<T, S: Shape>(&self, data: &[T]) -> Self::Data<T, S>
    where
        T: Clone;

    /// If the vector `vec` was allocated previously, this function can be used in order to reduce the amount of allocations, which may be faster than using a slice of `vec`.
    #[inline]
    #[cfg(not(feature = "no-std"))]
    fn alloc_from_vec<T, S: Shape>(&self, vec: Vec<T>) -> Self::Data<T, S>
    where
        T: Clone,
    {
        self.alloc_from_slice(&vec)
    }

    /// Allocates a pointer with the array provided by the `S:`[`Shape`] generic.
    /// By default, the array is flattened and then passed to [`Alloc::with_slice`].
    #[inline]
    fn alloc_from_array<T, S: Shape>(&self, array: S::ARR<T>) -> Self::Data<T, S>
    where
        T: Clone,
    {
        let stack_array = StackArray::<S, T>::from_array(array);
        self.alloc_from_slice(stack_array.flatten())
    }
}

impl<Mods> Alloc for CPU<Mods> {
    type Data<T, S: Shape> = CPUPtr<T>;

    fn alloc<T, S: Shape>(&self, mut len: usize, flag: AllocFlag) -> Self::Data<T, S> {
        assert!(len > 0, "invalid buffer len: 0");

        if S::LEN > len {
            len = S::LEN
        }

        CPUPtr::new_initialized(len, flag)
    }

    fn alloc_from_slice<T, S>(&self, data: &[T]) -> Self::Data<T, S>
    where
        S: Shape,
        T: Clone,
    {
        assert!(!data.is_empty(), "invalid buffer len: 0");
        assert!(S::LEN <= data.len(), "invalid buffer len: {}", data.len());

        let cpu_ptr = unsafe { CPUPtr::new(data.len(), AllocFlag::None) };
        let slice = unsafe { std::slice::from_raw_parts_mut(cpu_ptr.ptr, data.len()) };
        slice.clone_from_slice(data);

        cpu_ptr
    }

    fn alloc_from_vec<T, S: Shape>(&self, mut vec: Vec<T>) -> Self::Data<T, S>
    where
        T: Clone,
    {
        assert!(!vec.is_empty(), "invalid buffer len: 0");

        let ptr = vec.as_mut_ptr();
        let len = vec.len();
        core::mem::forget(vec);

        unsafe { CPUPtr::from_ptr(ptr, len, AllocFlag::None) }
    }
}
