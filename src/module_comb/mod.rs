mod features;
pub use features::*;

mod ptr_conv;
pub use ptr_conv::*;

mod modules;
pub use modules::*;

mod buffer;
pub use buffer::*;

mod location_id;
pub use location_id::*;

mod cache;
pub use cache::*;

use crate::{cpu::CPUPtr, flag::AllocFlag, Shape, StackArray};

#[cfg(test)]
pub fn location() -> &'static core::panic::Location<'static> {
    core::panic::Location::caller()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CPU<Mods> {
    modules: Mods,
}

pub trait Device {}

impl<Mods> Device for CPU<Mods> {}

pub trait Alloc: Sized {
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

pub trait Module<D> {
    type Module;

    fn new() -> Self::Module;
}

impl<SimpleMods> CPU<SimpleMods> {
    #[inline]
    pub fn new<NewMods>() -> CPU<NewMods>
    where
        SimpleMods: Module<CPU<SimpleMods>, Module = NewMods>,
    {
        CPU {
            modules: SimpleMods::new(),
        }
    }
}

pub trait Retriever: Alloc {
    #[track_caller]
    fn retrieve<T, S: Shape>(&self, len: usize) -> Buffer<T, Self, S>;
}

impl<Mods: Retrieve<Self>> Retriever for CPU<Mods> {
    #[inline]
    fn retrieve<T, S: Shape>(&self, len: usize) -> Buffer<T, Self, S> {
        let data = self.modules.retrieve::<T, S>(self, len);
        Buffer {
            data,
            device: Some(self),
            // id: LocationId::new()
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_init_new_buf() {
        let device = CPU::<Cached<Autograd<Base>>>::new();
        for _ in 0..100 {
            let buf = device.retrieve::<f32, ()>(10);
        }
    }

    use super::{Alloc, Autograd, Base, Cached, CachedModule, Module, Retrieve, Retriever, CPU};

    fn take_generic_dev<D: Retriever>(device: &D) {
        device.retrieve::<f32, ()>(10);
    }

    fn take_generic_dev_alloc<D: Alloc>(device: &D) {
        device.alloc::<f32, ()>(10, crate::flag::AllocFlag::None);
    }

    #[test]
    fn test_cpu_creation() {
        // take_generic_dev(&device);
        // CPU::<Cached<Autograd<Base>>>::new() -> select default type based on build time feature selection?
        let res = CPU::<Cached<Autograd<Base>>>::new();

        // let x: CachedModule<Base, _> = <Cached::<Base> as Module<CPU<Cached<Base>>>>::new();

        // let y: Autograd<CachedModule<Base, CPU<Base>>> = <Autograd<Cached<Base>> as Module<CPU<Base>, CachedModule<Base, CPU<Base>>>>::new();

        let res = CPU::<Autograd<Cached<Base>>>::new();

        take_generic_dev_alloc(&res);
        take_generic_dev(&res);
        // let device: CachedModule<Autograd<Base>, CPU<Base>> = Module::<Cached<Autograd<Base>>, CPU<Base>>::new();
    }

    #[test]
    fn test_address_of() {
        let mut vec = vec![1, 2, 3, 4];
        let ptr = vec.as_ptr();
        let ad = core::ptr::addr_of!(ptr) as usize;
        println!("ad: {ad}");
    }

    #[test]
    fn test_retrieve_unique_buf() {
        let device = CPU::<Cached<Base>>::new();
        let buf = device.retrieve::<f32, ()>(10);
        let buf1 = device.retrieve::<f32, ()>(10);
        assert_ne!(buf.data.ptr, buf1.data.ptr);
        let buf2 = device.retrieve::<f32, ()>(10);
        assert_ne!(buf.data.ptr, buf2.data.ptr);
        assert_ne!(buf1.data.ptr, buf2.data.ptr);
        let buf3 = device.retrieve::<f32, ()>(10);
        assert_ne!(buf2.data.ptr, buf3.data.ptr);
        assert_ne!(buf1.data.ptr, buf3.data.ptr);
        assert_ne!(buf.data.ptr, buf3.data.ptr);
    }
}
