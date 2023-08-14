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


#[cfg(test)]
mod tests {

    #[test]
    fn test_init_new_buf() {
        let device = CPU::<Cached<Autograd<Base>>>::new();
        for _ in 0..100 {
            let buf: super::Buffer<
                '_,
                f32,
                CPU<
                    CachedModule<
                        Autograd<CachedModule<Base, CPU<Cached<Autograd<Base>>>>>,
                        CPU<Cached<Autograd<Base>>>,
                    >,
                >,
                (),
            > = device.retrieve(10, ());
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
