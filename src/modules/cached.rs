use core::{cell::RefCell, marker::PhantomData};

use crate::{
    AddGradFn, AddLayer, AddOperation, Alloc, Buffer, Cache, Device, ExecNow, HasId, Module,
    OnDropBuffer, OnNewBuffer, Parents, PtrType, RemoveLayer, Retrieve, RunModule, Setup,
    ShallowCopy, Shape, WrappedData,
};

#[cfg(feature = "graph")]
use crate::{DeviceError, OptimizeMemGraph};

// creator struct, however =>
// TODO: could remove D generic and therefore CachedModule
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

impl<Mods: WrappedData, SD: Device> WrappedData for CachedModule<Mods, SD> {
    type Wrap<T, Base: HasId + PtrType> = Mods::Wrap<T, Base>;

    #[inline]
    fn wrap_in_base<T, Base: HasId + PtrType>(&self, base: Base) -> Self::Wrap<T, Base> {
        self.modules.wrap_in_base(base)
    }

    #[inline]
    fn wrapped_as_base<T, Base: HasId + PtrType>(wrap: &Self::Wrap<T, Base>) -> &Base {
        Mods::wrapped_as_base(wrap)
    }

    #[inline]
    fn wrapped_as_base_mut<T, Base: HasId + PtrType>(wrap: &mut Self::Wrap<T, Base>) -> &mut Base {
        Mods::wrapped_as_base_mut(wrap)
    }
}

impl<Mods: Module<D>, D: Device> Module<D> for Cached<Mods> {
    type Module = CachedModule<Mods::Module, D>;

    fn new() -> Self::Module {
        CachedModule {
            modules: Mods::new(),
            cache: RefCell::new(Cache::new()),
            pd: PhantomData,
        }
    }
}

// impl<Mods> OnDropBuffer for Cached<Mods> {}

// TODO: could remove D generic and therefore CachedModule
pub struct CachedModule<Mods, D: Device> {
    pub modules: Mods,
    pub cache: RefCell<Cache>,
    pub(crate) pd: PhantomData<D>,
}

impl<Mods: Setup<NewDev>, D: Device, NewDev> Setup<NewDev> for CachedModule<Mods, D> {
    #[inline]
    fn setup(device: &mut NewDev) -> crate::Result<()> {
        Mods::setup(device)
    }
}

impl<SD: Device, Mods: AddOperation> AddOperation for CachedModule<Mods, SD> {
    #[inline]
    fn ops_count(&self) -> usize {
        self.modules.ops_count()
    }

    fn add_op<Args: Parents<N>, const N: usize>(
        &self,
        mut args: Args,
        operation: fn(&mut Args) -> crate::Result<()>,
    ) -> crate::Result<()> {
        operation(&mut args)
    }
}

impl<D: Device, SD: Device, Mods: ExecNow<D>> ExecNow<D> for CachedModule<Mods, SD> {
    #[inline]
    fn exec_now(
        &self,
        device: &D,
        range_bounds: impl core::ops::RangeBounds<usize>,
    ) -> crate::Result<()> {
        self.modules.exec_now(device, range_bounds)
    }
}

impl<T, D: Device, Mods: OnNewBuffer<T, D, S>, SD: Device, S: Shape> OnNewBuffer<T, D, S>
    for CachedModule<Mods, SD>
{
    #[inline]
    fn on_new_buffer(&self, device: &D, new_buf: &Buffer<T, D, S>) {
        self.modules.on_new_buffer(device, new_buf)
    }
}

impl<Mods: OnDropBuffer, SD: Device> OnDropBuffer for CachedModule<Mods, SD> {
    #[inline]
    fn on_drop_buffer<T, D: Device, S: Shape>(&self, device: &D, buf: &Buffer<T, D, S>) {
        self.modules.on_drop_buffer(device, buf)
    }
}

// TODO: a more general OnDropBuffer => "Module"
impl<T, Mods, D, SimpleDevice, S: Shape> Retrieve<D, T, S> for CachedModule<Mods, SimpleDevice>
where
    Mods: Retrieve<D, T, S>,
    D: Device + 'static,
    D::Base<T, S>: ShallowCopy + 'static,
    SimpleDevice: Device,
{
    #[inline]
    unsafe fn retrieve<const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        _parents: impl Parents<NUM_PARENTS>,
    ) -> Self::Wrap<T, D::Base<T, S>>
    where
        D: Alloc<T>,
    {
        self.wrap_in_base(self.cache.borrow_mut().get(device, len, || ()))
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
impl<Mods: crate::HasAutograd, SD: Device> crate::HasAutograd for CachedModule<Mods, SD> {}

#[cfg(feature = "autograd")]
impl<Mods: crate::TapeActions, SD: Device> crate::TapeActions for CachedModule<Mods, SD> {
    #[inline]
    unsafe fn tape(&self) -> Option<&super::Tape> {
        self.modules.tape()
    }

    #[inline]
    unsafe fn tape_mut(&self) -> Option<&mut super::Tape> {
        self.modules.tape_mut()
    }

    #[inline]
    unsafe fn gradients(&self) -> Option<&crate::Gradients> {
        self.modules.gradients()
    }

    #[inline]
    unsafe fn gradients_mut(&self) -> Option<&mut crate::Gradients> {
        self.modules.gradients_mut()
    }
}

impl<CurrentMods, SD: Device> AddLayer<CurrentMods, SD> for Cached<()> {
    type Wrapped = crate::CachedModule<CurrentMods, SD>;

    #[inline]
    fn wrap_layer(inner_mods: CurrentMods) -> Self::Wrapped {
        crate::CachedModule {
            modules: inner_mods,
            cache: Default::default(),
            pd: core::marker::PhantomData,
        }
    }
}

impl<Mods, SD: Device> RemoveLayer<Mods> for CachedModule<Mods, SD> {
    #[inline]
    fn inner_mods(self) -> Mods {
        self.modules
    }
}

impl<Mods: AddGradFn, D: Device> AddGradFn for CachedModule<Mods, D> {
    #[inline]
    fn add_grad_fn<Args: Parents<N> + crate::UpdateArgs, const N: usize>(
        &self,
        args: Args,
        op: fn(&mut Args) -> crate::Result<()>,
    ) {
        self.modules.add_grad_fn(args, op)
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

impl<Mods: RunModule<D>, D, SD: Device> RunModule<D> for CachedModule<Mods, SD> {
    #[inline]
    fn run(&self, _device: &D) -> crate::Result<()> {
        self.modules.run(_device)
    }
}

#[cfg(feature = "graph")]
impl<Mods: OptimizeMemGraph, SD: Device> OptimizeMemGraph for CachedModule<Mods, SD> {
    fn optimize_mem_graph<D: 'static>(
        &self,
        _device: &D,
        graph_translator: Option<&crate::GraphTranslator>,
        //cache_traces: Option<&[crate::TranslatedCacheTrace]>,
    ) -> crate::Result<()> {
        let graph_translator = graph_translator.ok_or(DeviceError::MissingCacheTraces)?;
        let cache_traces = graph_translator
            .to_hash_location_cache_traces(graph_translator.opt_graph.cache_traces());

        let mut cache = self.cache.borrow_mut();
        for cache_trace in cache_traces {
            let used_to_replace = cache
                .nodes
                .get(&cache_trace.cache_idx)
                .ok_or(DeviceError::GraphOptimization)?
                .clone();

            for to_replace in &cache_trace.use_cache_idxs {
                cache.nodes.insert(*to_replace, used_to_replace.clone());
            }
        }
        Ok(())
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
/// fn add_bufs<Mods: Retrieve<CPU<Mods>, f32>>(device: &CPU<Mods>) -> Buffer<f32, CPU<Mods>, ()> {
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
    fn add_bufs<Mods: Retrieve<CPU<Mods>, f32>>(device: &CPU<Mods>) -> Buffer<f32, CPU<Mods>, ()> {
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
    fn add_bufs_tracked<Mods: Retrieve<CPU<Mods>, f32>>(
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

    #[cfg(feature = "cpu")]
    #[test]
    fn test_cached_return_retrieve() {
        // invalid!
        let _x = {
            let device = CPU::<Cached<Base>>::new();
            // let buf: Buffer<f32, _> = device.retrieve(10, ());
            unsafe { Retrieve::<_, f32, ()>::retrieve(&device.modules, &device, 10, ()) }
        };
    }
}
