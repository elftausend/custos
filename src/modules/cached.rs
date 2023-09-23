use core::{cell::RefCell, marker::PhantomData};

use crate::{
    flag::AllocFlag, Alloc, Buffer, Cache, Device, Module, Parents, PtrConv, Retrieve, Setup, Shape,
};

// creator struct
#[derive(Debug, PartialEq, Eq, Default)]
pub struct Cached<Mods> {
    pd: PhantomData<Mods>,
}

/*impl<Mods, D> Retrieve<D> for Cached<Mods> {
    fn retrieve<T, S: Shape>(&self, device: &D, len: usize) -> <D>::Data<T, S>
    where
        D: Alloc
    {
        todo!()
    }
}*/

impl<Mods: Module<D>, D: Device> Module<D> for Cached<Mods> {
    type Module = CachedModule<Mods::Module, D>;

    fn new() -> Self::Module {
        CachedModule {
            modules: Mods::new(),
            cache: RefCell::new(Cache {
                nodes: Default::default(),
            }),
        }
    }
}

// impl<Mods> OnDropBuffer for Cached<Mods> {}

pub struct CachedModule<Mods, D: Device> {
    pub modules: Mods,
    pub cache: RefCell<Cache<D>>,
}

impl<Mods: Setup<NewDev>, D: Device, NewDev> Setup<NewDev> for CachedModule<Mods, D> {
    #[inline]
    fn setup(device: &mut NewDev) -> crate::Result<()> {
        Mods::setup(device)
    }
}

// TODO: a more general OnDropBuffer => "Module"
impl<T, Mods, D, S, SimpleDevice> Retrieve<D, T, S> for CachedModule<Mods, SimpleDevice>
where
    S: Shape,
    Mods: Retrieve<D, T, S>,
    D: Device + PtrConv<SimpleDevice>,
    SimpleDevice: Device + PtrConv<D>,
{
    #[inline]
    fn retrieve<const NUM_PARENTS: usize>(
        &self,
        device: &D,
        _len: usize,
        _parents: impl Parents<NUM_PARENTS>,
        alloc_fn: impl FnOnce(&D, AllocFlag) -> D::Data<T, S>,
    ) -> D::Data<T, S>
    where
        D: Alloc<T>,
    {
        self.cache.borrow_mut().get(device, alloc_fn)
    }

    #[inline]
    fn on_retrieve_finish(&self, retrieved_buf: &Buffer<T, D, S>)
    where
        D: Alloc<T>,
    {
        self.modules.on_retrieve_finish(retrieved_buf)
    }
}

#[cfg(feature = "autograd")]
impl<Mods: crate::TapeActions, SD: Device> crate::TapeActions for CachedModule<Mods, SD> {
    #[inline]
    fn tape(&self) -> Option<core::cell::Ref<super::Tape>> {
        self.modules.tape()
    }

    #[inline]
    fn tape_mut(&self) -> Option<core::cell::RefMut<super::Tape>> {
        self.modules.tape_mut()
    }
}

impl<Mods: crate::UseGpuOrCpu, D: Device> crate::UseGpuOrCpu for CachedModule<Mods, D> {
    #[inline]
    fn use_cpu_or_gpu(
        &self,
        location: crate::HashLocation<'static>,
        input_lengths: &[usize],
        cpu_op: impl FnMut(),
        gpu_op: impl FnMut(),
    ) -> crate::GpuOrCpuInfo {
        self.modules
            .use_cpu_or_gpu(location, input_lengths, cpu_op, gpu_op)
    }
}

#[macro_export]
macro_rules! debug_assert_tracked {
    () => {
        #[cfg(debug_assertions)]
        {
            let location = core::panic::Location::caller();
            assert_ne!(
                (file!(), line!(), column!()),
                (location.file(), location.line(), location.column()),
                "Function and operation must be annotated with `#[track_caller]`."
            );
        }
    };
}

/// This macro is nothing but a mechanism to ensure that the specific operation is annotated with `#[track_caller]`.
/// If the operation is not annotated with `#[track_caller]`, then the macro will cause a panic (in debug mode).
///
/// This macro turns the device, length and optionally type information into the following line of code:
/// ## From:
/// ```ignore
/// retrieve!(device, 10, f32)
/// ```
/// ## To:
/// ```ignore
/// custos::debug_assert_tracked!();
/// device.retrieve::<f32, ()>(10)
/// ```
///
/// If you ensure that the operation is annotated with `#[track_caller]`, then you can just write the following:
/// ```ignore
/// device.retrieve::<f32, ()>(10)
/// ```
///
/// # Example
/// Operation is not annotated with `#[track_caller]` and therefore will panic:
/// ```should_panic
/// use custos::{retrieve, CPU, Retriever, Buffer, Retrieve, Cached, Base};
///
/// fn add_bufs<Mods: Retrieve<CPU<Mods>, f32, ()>>(device: &CPU<Mods>) -> Buffer<f32, CPU<Mods>, ()> {
///     retrieve!(device, 10, ())
/// }
///
/// let device = CPU::<Cached<Base>>::new();
/// add_bufs(&device);
/// ```
/// Operation is annotated with `#[track_caller]`:
/// ```
/// use custos::{Dim1, retrieve, CPU, Retriever, Buffer, Retrieve, Cached, Base};
///
/// #[track_caller]
/// fn add_bufs<Mods: Retrieve<CPU<Mods>, f32, Dim1<30>>>(device: &CPU<Mods>) -> Buffer<f32, CPU<Mods>, Dim1<30>> {
///     retrieve!(device, 10, ())
/// }
///
/// let device = CPU::<Cached<Base>>::new();
/// add_bufs(&device);
/// ```
#[macro_export]
macro_rules! retrieve {
    ($device:ident, $len:expr, $parents:expr) => {{
        $crate::debug_assert_tracked!();
        $device.retrieve($len, $parents)
    }}; /*($device:ident, $len:expr, $dtype:ty, ) => {{
            $crate::debug_assert_tracked!();
            $device.retrieve::<$dtype, ()>($len)
        }};
        ($device:ident, $len:expr, $dtype:ty, $shape:ty) => {{
            $crate::debug_assert_tracked!();
            $device.retrieve::<$dtype, $shape>($len)
        }};*/
}

#[cfg(test)]
mod tests {
    use core::{panic::Location, ptr::addr_of};

    // crate::modules
    use crate::{location, Base, Buffer, Retrieve, Retriever, CPU};

    use super::Cached;

    // forgot to add track_caller
    #[cfg(debug_assertions)]
    fn add_bufs<Mods: Retrieve<CPU<Mods>, f32, ()>>(
        device: &CPU<Mods>,
    ) -> Buffer<f32, CPU<Mods>, ()> {
        retrieve!(device, 10, ())
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn test_forgot_track_caller_runtime_detection() {
        let device = CPU::<Cached<Base>>::new();

        let _out = add_bufs(&device);
        let _out = add_bufs(&device);
    }

    #[track_caller]
    fn add_bufs_tracked<Mods: Retrieve<CPU<Mods>, f32, ()>>(
        device: &CPU<Mods>,
    ) -> Buffer<f32, CPU<Mods>, ()> {
        retrieve!(device, 10, ())
    }

    #[test]
    fn test_added_track_caller() {
        let device = CPU::<Cached<Base>>::new();

        let _out = add_bufs_tracked(&device);
        let _out = add_bufs_tracked(&device);
    }

    #[test]
    fn test_location_ref_unique() {
        let ptr = location();
        let ptr1 = location();
        // bad
        assert_ne!(addr_of!(ptr), addr_of!(ptr1));
    }

    #[track_caller]
    fn location_tracked() -> &'static Location<'static> {
        Location::caller()
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_location_file_ptr_unique() {
        let ptr = location();
        let ptr1 = location();
        // good
        assert_eq!(ptr.file().as_ptr(), ptr1.file().as_ptr());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_location_file_tracked_ptr_unique() {
        let ptr = location_tracked();
        let ptr1 = location_tracked();
        // good
        assert_eq!(ptr.file().as_ptr(), ptr1.file().as_ptr());
    }

    #[test]
    fn test_location_with_different_file_location_ptr_unique() {
        let ptr = location_tracked();
        let ptr1 = location();
        // good
        assert_ne!(ptr.file().as_ptr(), ptr1.file().as_ptr());
    }
}
