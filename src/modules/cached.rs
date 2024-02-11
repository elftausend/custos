use core::{
    cell::{Cell, RefCell},
    marker::PhantomData,
};

use crate::{
    AddGradFn, AddLayer, AddOperation, Alloc, Buffer, Cache, CachedBuffers, Cursor, Device,
    ExecNow, HasId, IsShapeIndep, Module, OnDropBuffer, OnNewBuffer, Parents, PtrType, RemoveLayer,
    Retrieve, RunModule, Setup, ShallowCopy, Shape, UniqueId, WrappedData,
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
            cursor: Default::default(),
        }
    }
}

// impl<Mods> OnDropBuffer for Cached<Mods> {}

// TODO: could remove D generic and therefore CachedModule
pub struct CachedModule<Mods, D: Device> {
    pub modules: Mods,
    pub cache: RefCell<Cache>,
    pub(crate) pd: PhantomData<D>,
    cursor: Cell<usize>, // would move this to `Cache`, however -> RefCell; TODO: maybe add a Cursor Module
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

    #[inline]
    fn set_lazy_enabled(&self, enabled: bool) {
        self.modules.set_lazy_enabled(enabled)
    }

    #[inline]
    fn is_lazy_enabled(&self) -> bool {
        self.modules.is_lazy_enabled()
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

impl<T, D, Mods, SD, S> OnNewBuffer<T, D, S> for CachedModule<Mods, SD>
where
    T: 'static,
    D: Device + IsShapeIndep + 'static,
    Mods: OnNewBuffer<T, D, S>,
    D::Data<T, S>: ShallowCopy,
    SD: Device,
    S: Shape,
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
    T: 'static,
    Mods: Retrieve<D, T, S>,
    D: Device + IsShapeIndep + Cursor + 'static,
    D::Base<T, S>: ShallowCopy + 'static,
    D::Data<T, S>: ShallowCopy + 'static,
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
        self.wrap_in_base(
            self.cache
                .borrow_mut()
                .get(device, len, |_cursor, _base| {}),
        )
    }

    #[inline]
    fn on_retrieve_finish(&self, retrieved_buf: &Buffer<T, D, S>)
    where
        D: Alloc<T>,
    {
        self.modules.on_retrieve_finish(retrieved_buf)
    }
}

impl<Mods, SD: Device> Cursor for CachedModule<Mods, SD> {
    #[inline]
    fn cursor(&self) -> usize {
        self.cursor.get()
    }

    #[inline]
    unsafe fn set_cursor(&self, cursor: usize) {
        self.cursor.set(cursor)
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
            cursor: Default::default(),
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

    #[inline]
    fn set_grad_enabled(&self, enabled: bool) {
        self.modules.set_grad_enabled(enabled)
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
    ) -> crate::Result<()> {
        let graph_translator = graph_translator.ok_or(DeviceError::MissingCacheTraces)?;
        let cache_traces =
            graph_translator.to_cursor_cache_traces(graph_translator.opt_graph.cache_traces());

        let mut cache = self.cache.borrow_mut();
        for cache_trace in cache_traces {
            let used_to_replace = cache
                .nodes
                .get(&(cache_trace.cache_idx as UniqueId))
                .ok_or(DeviceError::GraphOptimization)?
                .clone();

            for to_replace in &cache_trace.use_cache_idxs {
                if cache
                    .nodes
                    .get(&(*to_replace as UniqueId))
                    .unwrap()
                    .type_id()
                    != used_to_replace.type_id()
                {
                    continue;
                }
                cache
                    .nodes
                    .insert(*to_replace as UniqueId, used_to_replace.clone());
            }
        }
        Ok(())
    }
}

#[macro_export]
#[deprecated]
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

impl<Mods: OnDropBuffer, D: Device> CachedBuffers for CachedModule<Mods, D> {
    #[inline]
    unsafe fn buffers_mut(
        &self,
    ) -> Option<core::cell::RefMut<crate::Buffers<Box<dyn crate::BoxedShallowCopy>>>> {
        // Use the stored buffers in autograd module -> optimizing isn't possible anyway
        None
    }
}

#[cfg(test)]
mod tests {
    use core::{panic::Location, ptr::addr_of};

    // crate::modules
    use crate::{location, Base, Buffer, Retrieve, Retriever, CPU};

    use super::Cached;
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

    #[track_caller]
    #[cfg(feature = "cpu")]
    fn level1<Mods: crate::Retrieve<CPU<Mods>, f32, ()>>(device: &CPU<Mods>) {
        let _buf: Buffer<f32, _> = device.retrieve(10, ());
        level2(device);
        level2(device);
        level3(device);
    }

    #[track_caller]
    #[cfg(feature = "cpu")]
    fn level3<Mods: crate::Retrieve<CPU<Mods>, f32, ()>>(device: &CPU<Mods>) {
        level2(device);
    }

    #[track_caller]
    #[cfg(feature = "cpu")]
    fn level2<Mods: crate::Retrieve<CPU<Mods>, f32, ()>>(device: &CPU<Mods>) {
        let buf: Buffer<f32, _> = device.retrieve(20, ());
        location();
        assert_eq!(buf.len(), 20);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_multi_level_retrieve() {
        let device = CPU::<Cached<Base>>::new();
        level1(&device);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_add_cached_layer() {
        use crate::{ApplyFunction, Combiner, UnaryElementWiseMayGrad};

        let device = CPU::<Base>::new();
        let buf_base: Buffer<f32, _> = device.retrieve(10, ());

        let device = device.add_layer::<Cached<()>>();

        for _ in 0..10 {
            let buf: Buffer<f32, _> = device.retrieve(10, &buf_base);

            for (base, cached) in buf_base.iter().zip(buf.iter()) {
                assert_eq!(base, cached);
            }

            let _x = device.apply_fn(&buf_base, |x| x.exp());
            assert_eq!(buf.len(), buf_base.len());
        }

        let buf_base = buf_base.to_device_type(&device);
        for _ in 0..10 {
            let buf: Buffer<f32, _> = device.retrieve(10, &buf_base);

            for (base, cached) in buf_base.iter().zip(buf.iter()) {
                assert_eq!(base, cached);
            }

            let _x = device.unary_ew(&buf_base, |x| x.exp(), |x| x.exp());
            assert_eq!(buf.len(), buf_base.len());
        }
    }
}
