use crate::{flag::AllocFlag, PtrType, Shape, StackArray};

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

mod devices;
pub use devices::*;

mod id;
pub use id::*;

mod hooks;
pub use hooks::*;

mod op_traits;
pub use op_traits::*;

mod apply_fn;
pub use apply_fn::*;

#[cfg(test)]
pub fn location() -> &'static core::panic::Location<'static> {
    core::panic::Location::caller()
}

pub trait Alloc: Sized {
    type Data<T, S: Shape>: HasId + PtrType;

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

pub trait Module<D> {
    type Module;

    fn new() -> Self::Module;
}

/// Used for modules that should affect the device.
pub trait Setup<D> {
    #[inline]
    fn setup(_device: &mut D) {}
}

pub trait Retriever: Device {
    #[track_caller]
    fn retrieve<T, S, const NUM_PARENTS: usize>(
        &self,
        len: usize,
        parents: impl Parents<NUM_PARENTS>,
    ) -> Buffer<T, Self, S>
    where
        T: 'static,
        S: Shape;
}

/// Devices that can access the main memory / RAM of the host.
pub trait MainMemory: Device {
    /// Returns the respective immutable host memory pointer
    fn as_ptr<T, S: Shape>(ptr: &Self::Data<T, S>) -> *const T;
    /// Returns the respective mutable host memory pointer
    fn as_ptr_mut<T, S: Shape>(ptr: &mut Self::Data<T, S>) -> *mut T;
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_init_new_buf() {
        let device = CPU::<Cached<Autograd<Base>>>::new();
        for _ in 0..100 {
            let buf: super::Buffer<'_, f32, CPU<CachedModule<Autograd<CachedModule<Base, CPU<Cached<Autograd<Base>>>>>, CPU<Cached<Autograd<Base>>>>>, ()> = device.retrieve(10, ());
        }
    }

    use super::{Alloc, Autograd, Base, Cached, CachedModule, Module, Retrieve, Retriever, CPU};

    fn take_generic_dev<D: Retriever>(device: &D) {
        device.retrieve::<f32, (), 0>(10, ());
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
        let buf = device.retrieve::<f32, (), 0>(10, ());
        let buf1 = device.retrieve::<f32, (), 0>(10, ());
        assert_ne!(buf.data.ptr, buf1.data.ptr);
        let buf2 = device.retrieve::<f32, (), 0>(10, ());
        assert_ne!(buf.data.ptr, buf2.data.ptr);
        assert_ne!(buf1.data.ptr, buf2.data.ptr);
        let buf3 = device.retrieve::<f32, (), 0>(10, ());
        assert_ne!(buf2.data.ptr, buf3.data.ptr);
        assert_ne!(buf1.data.ptr, buf3.data.ptr);
        assert_ne!(buf.data.ptr, buf3.data.ptr);
    }
}
