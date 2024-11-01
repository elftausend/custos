use core::{
    cell::{Cell, RefCell, RefMut},
    marker::PhantomData,
};

use crate::{
    AddGradFn, AddLayer, AddOperation, Alloc, Buffer, Cache, CachedBuffers, CowMut, Cursor, Device,
    ExecNow, FastCache, Guard, HasId, HasModules, IsBasePtr, IsShapeIndep, LockInfo, Module,
    OnDropBuffer, OnNewBuffer, Parents, PtrType, RemoveLayer, ReplaceBuf, Retrieve, RunModule,
    SetOpHint, Setup, ShallowCopy, Shape, State, UniqueId, Unit, WrappedData,
};

#[cfg(feature = "graph")]
use crate::{DeviceError, Optimize};

// creator struct, however =>
// TODO: could remove D generic and therefore CachedModule
#[derive(Debug, PartialEq, Eq, Default)]
pub struct Cached<Mods, CacheType = FastCache> {
    pd: PhantomData<Mods>,
    cache_type: PhantomData<CacheType>,
}

impl<CacheType, Mods: WrappedData, SD: Device> WrappedData for CachedModule<Mods, SD, CacheType> {
    type Wrap<'a, T: Unit, Base: IsBasePtr> = Guard<'a, Mods::Wrap<'static, T, Base>>;

    #[inline]
    fn wrap_in_base<'a, T: Unit, Base: IsBasePtr>(&self, base: Base) -> Self::Wrap<'a, T, Base> {
        Guard::new(CowMut::Owned(self.modules.wrap_in_base(base)))
    }

    #[inline]
    fn wrapped_as_base<'a, 'b, T: Unit, Base: IsBasePtr>(
        wrap: &'b Self::Wrap<'a, T, Base>,
    ) -> &'b Base {
        Mods::wrapped_as_base(wrap)
    }

    #[inline]
    fn wrapped_as_base_mut<'a, 'b, T: Unit, Base: IsBasePtr>(
        wrap: &'b mut Self::Wrap<'a, T, Base>,
    ) -> &'b mut Base {
        Mods::wrapped_as_base_mut(wrap)
    }
}

impl<'a, CacheType, Mods: Module<'a, D>, D: Device + 'a> Module<'a, D> for Cached<Mods, CacheType>
where
    CacheType: Default,
{
    type Module = CachedModule<Mods::Module, D, CacheType>;

    fn new() -> Self::Module {
        CachedModule {
            modules: Mods::new(),
            cache: RefCell::new(CacheType::default()),
            cache3: Default::default(),
            pd: PhantomData,
            cursor: Default::default(),
        }
    }
}

// impl<Mods> OnDropBuffer for Cached<Mods> {}

// TODO: could remove D generic and therefore CachedModule
pub struct CachedModule<Mods, D: Device, CacheType = FastCache> {
    pub modules: Mods,
    pub cache: RefCell<CacheType>,
    pub cache3: crate::LockedMap<u64, Box<dyn core::any::Any>>,
    pub(crate) pd: PhantomData<D>,
    cursor: Cell<usize>, // would move this to `Cache`, however -> RefCell; TODO: maybe add a Cursor Module
}

impl<CacheType, Mods: Setup<NewDev>, D: Device, NewDev> Setup<NewDev>
    for CachedModule<Mods, D, CacheType>
{
    #[inline]
    fn setup(device: &mut NewDev) -> crate::Result<()> {
        Mods::setup(device)
    }
}

impl<CacheType, SD: Device, Mods: AddOperation> AddOperation for CachedModule<Mods, SD, CacheType> {
    #[inline]
    fn add_op<Args: Parents<N> + crate::AnyOp, const N: usize>(
        &self,
        args: Args,
        op: impl for<'b> Fn(Args::Replicated<'b>) -> crate::Result<()> + 'static,
    ) -> crate::Result<()> {
        self.modules.add_op(args, op)
    }

    #[inline]
    fn ops_count(&self) -> usize {
        self.modules.ops_count()
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

impl<CacheType, T, Mods: SetOpHint<T>, SD: Device> SetOpHint<T>
    for CachedModule<Mods, SD, CacheType>
{
    #[inline]
    fn set_op_hint(&self, op_hint: crate::op_hint::OpHint<T>) {
        self.modules.set_op_hint(op_hint)
    }
}

impl<CacheType, D: Device, SD: Device, Mods: ExecNow<D>> ExecNow<D>
    for CachedModule<Mods, SD, CacheType>
{
    #[inline]
    fn exec_now(
        &self,
        device: &D,
        range_bounds: impl core::ops::RangeBounds<usize>,
    ) -> crate::Result<()> {
        self.modules.exec_now(device, range_bounds)
    }
}

impl<'a, CacheType, T, D, Mods, SD, S> OnNewBuffer<'a, T, D, S>
    for CachedModule<Mods, SD, CacheType>
where
    T: Unit + 'static,
    D: Device + IsShapeIndep + 'static,
    Mods: OnNewBuffer<'a, T, D, S>,
    D::Data<'a, T, S>: ShallowCopy,
    SD: Device,
    S: Shape,
{
    #[inline]
    unsafe fn on_new_buffer(&self, device: &'a D, new_buf: &Buffer<'a, T, D, S>) {
        self.modules.on_new_buffer(device, new_buf)
    }
}

impl<CacheType, Mods: OnDropBuffer, SD: Device> OnDropBuffer for CachedModule<Mods, SD, CacheType> {
    #[inline]
    fn on_drop_buffer<T: Unit, D: Device, S: Shape>(&self, device: &D, buf: &Buffer<T, D, S>) {
        self.modules.on_drop_buffer(device, buf)
    }
}

impl<'a, CacheType, Mods, SimpleDevice> CachedModule<Mods, SimpleDevice, CacheType>
where
    Mods: WrappedData,
    SimpleDevice: Device,
{
    pub fn get<D, T, S>(
        &'a self,
        id: u64,
    ) -> State<Guard<'a, Mods::Wrap<'static, T, D::Base<T, S>>>>
    where
        D: Device,
        T: 'static,
        S: Shape,
    {
        let entry = self.cache3.get_mut(&id)?;
        let entry = RefMut::map(entry, |x| {
            x.downcast_mut::<Mods::Wrap<'static, T, D::Base<T, S>>>()
                .unwrap()
        });
        Ok(Guard::new(CowMut::BorrowedMut(entry)))
    }
}

// TODO: a more general OnDropBuffer => "Module"
impl<'a, CacheType, T, Mods, D, SimpleDevice, S: Shape> Retrieve<'a, D, T, S>
    for CachedModule<Mods, SimpleDevice, CacheType>
where
    T: Unit + 'static,
    Mods: Retrieve<'static, D, T, S>,
    D: Device + IsShapeIndep + Cursor,
    D::Base<T, S>: 'static,
    SimpleDevice: Device,
    CacheType: Cache,
{
    #[inline]
    unsafe fn retrieve_entry<const NUM_PARENTS: usize>(
        &'a self,
        device: &D,
        len: usize,
        _parents: &impl Parents<NUM_PARENTS>,
    ) -> crate::Result<Self::Wrap<'a, T, D::Base<T, S>>>
    where
        D: Alloc<T>,
    {
        let id = device.cursor() as UniqueId;
        match self.get::<D, T, S>(id) {
            Ok(out) => Ok(out),
            Err(state) => match state {
                LockInfo::Locked => panic!("Locked!!"),
                LockInfo::None => {
                    self.cache3
                        .insert(id, Box::new(self.modules.retrieve(device, len, _parents)));
                    Ok(self.get::<D, T, S>(id).unwrap())
                }
            },
        }
        // let retrieved = Ok(self.wrap_in_base(self.cache.borrow_mut().get(
        //     device,
        //     device.cursor() as UniqueId,
        //     len,
        //     |_cursor, _base| {},
        // )));
        // unsafe { device.bump_cursor() };
        // retrieved
    }

    #[inline]
    fn on_retrieve_finish<const NUM_PARENTS: usize>(
        &self,
        len: usize,
        parents: impl Parents<NUM_PARENTS>,
        retrieved_buf: &Buffer<T, D, S>,
    ) where
        D: Alloc<T>,
    {
        self.modules.on_retrieve_finish(len, parents, retrieved_buf)
    }

    unsafe fn retrieve<const NUM_PARENTS: usize>(
        &self,
        _device: &D,
        _len: usize,
        _parents: &impl Parents<NUM_PARENTS>,
    ) -> crate::Result<Self::Wrap<'a, T, <D>::Base<T, S>>>
    where
        S: Shape,
        D: Device + Alloc<T>,
    {
        panic!("Modules retrieve calls are in the wrong order. Cached module requires to be called via 'retrieve_entry'")
    }
}

impl<CacheType, Mods, SD: Device> Cursor for CachedModule<Mods, SD, CacheType> {
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
impl<CacheType, Mods: crate::HasAutograd, SD: Device> crate::HasAutograd
    for CachedModule<Mods, SD, CacheType>
{
}

#[cfg(feature = "autograd")]
impl<'dev, CacheType, Mods: crate::TapeActions<'dev>, SD: Device> crate::TapeActions<'dev>
    for CachedModule<Mods, SD, CacheType>
{
    #[inline]
    unsafe fn tape(&self) -> Option<&super::Tape<'dev>> {
        self.modules.tape()
    }

    #[inline]
    unsafe fn tape_mut(&self) -> Option<&mut super::Tape<'dev>> {
        self.modules.tape_mut()
    }
}

#[cfg(feature = "autograd")]
impl<CacheType, Mods: crate::GradActions, SD: Device> crate::GradActions
    for CachedModule<Mods, SD, CacheType>
{
    unsafe fn grad<
        'a,
        T: 'static,
        D: Device + Alloc<T> + crate::ZeroGrad<T> + 'static,
        S: Shape,
    >(
        &self,
        device: &'a D,
        buf: &Buffer<'a, T, D, S>,
    ) -> &Buffer<'a, T, D, S> {
        self.modules.grad(device, buf)
    }

    unsafe fn grad_mut<
        'a,
        T: 'static,
        D: Device + Alloc<T> + crate::ZeroGrad<T> + 'static,
        S: Shape,
    >(
        &self,
        device: &'a D,
        buf: &Buffer<'a, T, D, S>,
    ) -> &mut Buffer<'a, T, D, S> {
        self.modules.grad_mut(device, buf)
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

impl<CacheType, CurrentMods, SD: Device> AddLayer<CurrentMods, SD> for Cached<(), CacheType> {
    type Wrapped = crate::CachedModule<CurrentMods, SD>;

    #[inline]
    fn wrap_layer(inner_mods: CurrentMods) -> Self::Wrapped {
        crate::CachedModule {
            modules: inner_mods,
            cache: Default::default(),
            cache3: Default::default(),
            pd: core::marker::PhantomData,
            cursor: Default::default(),
        }
    }
}

impl<CacheType, Mods, SD: Device> RemoveLayer<Mods> for CachedModule<Mods, SD, CacheType> {
    #[inline]
    fn inner_mods(self) -> Mods {
        self.modules
    }
}

impl<CacheType, Mods: AddGradFn, D: Device> AddGradFn for CachedModule<Mods, D, CacheType> {
    #[inline]
    fn add_grad_fn<Args: Parents<N> + crate::AnyOp, const N: usize>(
        &self,
        args: Args,
        op: impl for<'b> Fn(Args::Replicated<'b>) -> crate::Result<()> + 'static,
    ) {
        self.modules.add_grad_fn(args, op)
    }

    #[inline]
    fn set_grad_enabled(&self, enabled: bool) {
        self.modules.set_grad_enabled(enabled)
    }
}

impl<CacheType, Mods: crate::UseGpuOrCpu, D: Device> crate::UseGpuOrCpu
    for CachedModule<Mods, D, CacheType>
{
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

    #[inline]
    fn set_fork_enabled(&self, _enabled: bool) {
        self.modules.set_fork_enabled(_enabled)
    }

    #[inline]
    fn is_fork_enabled(&self) -> bool {
        self.modules.is_fork_enabled()
    }
}

impl<CacheType, Mods: RunModule<D>, D, SD: Device> RunModule<D>
    for CachedModule<Mods, SD, CacheType>
{
    #[inline]
    fn run(&self, _device: &D) -> crate::Result<()> {
        self.modules.run(_device)
    }
}

#[cfg(feature = "graph")]
impl<Mods: Optimize, SD: Device> Optimize for CachedModule<Mods, SD, FastCache> {
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

    #[inline]
    fn unary_fusing<D: 'static>(
        &self,
        _device: &D,
        _graph_translator: Option<&crate::modules::GraphTranslator>,
    ) -> crate::Result<()> {
        Err(DeviceError::UnaryFusingUnsupported.into())
    }
}

impl<CacheType, Mods: OnDropBuffer, D: Device> CachedBuffers for CachedModule<Mods, D, CacheType> {
    #[inline]
    unsafe fn buffers_mut(
        &self,
    ) -> Option<core::cell::RefMut<crate::Buffers<Box<dyn crate::BoxedShallowCopy>>>> {
        // Use the stored buffers in autograd module -> optimizing isn't possible anyway
        None
    }
}

impl<CacheType, Mods, D, T, S, SD> ReplaceBuf<T, D, S> for CachedModule<Mods, SD, CacheType>
where
    T: Unit,
    Mods: ReplaceBuf<T, D, S>,
    D: Device,
    S: Shape,
    SD: Device,
{
    #[inline]
    fn replace_buf<'a, 'c>(&'c self, buffer: &'c Buffer<'a, T, D, S>) -> &'c Buffer<'a, T, D, S> {
        self.modules.replace_buf(buffer)
    }
}

impl<CacheType, Mods, D: Device> HasModules for CachedModule<Mods, D, CacheType> {
    type Mods = Mods;

    #[inline]
    fn modules(&self) -> &Self::Mods {
        &self.modules
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
        {
            let device = CPU::<Cached<Base>>::new();
            // let buf: Buffer<f32, _> = device.retrieve(10, ());
            unsafe { Retrieve::<_, f32, ()>::retrieve_entry(&device.modules, &device, 10, &()) };
        };
    }

    #[track_caller]
    #[cfg(feature = "cpu")]
    fn level1<'a, Mods: crate::Retrieve<'a, CPU<Mods>, f32, ()>>(device: &'a CPU<Mods>) {
        let _buf: Buffer<f32, _> = device.retrieve(10, ()).unwrap();
        level2(device);
        level2(device);
        level3(device);
    }

    #[track_caller]
    #[cfg(feature = "cpu")]
    fn level3<'a, Mods: crate::Retrieve<'a, CPU<Mods>, f32, ()>>(device: &'a CPU<Mods>) {
        level2(device);
    }

    #[track_caller]
    #[cfg(feature = "cpu")]
    fn level2<'a, Mods: crate::Retrieve<'a, CPU<Mods>, f32, ()>>(device: &'a CPU<Mods>) {
        let buf: Buffer<f32, _> = device.retrieve(20, ()).unwrap();
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
        let buf_base: Buffer<f32, _> = device.retrieve(10, ()).unwrap();

        let device = device.add_layer::<Cached<()>>();

        for _ in 0..10 {
            let buf: Buffer<f32, _> = device.retrieve(10, &buf_base).unwrap();

            for (base, cached) in buf_base.iter().zip(buf.iter()) {
                assert_eq!(base, cached);
            }

            let _x = device.apply_fn(&buf_base, |x| x.exp());
            assert_eq!(buf.len(), buf_base.len());
        }

        let buf_base = buf_base.to_device_type(&device);
        for _ in 0..10 {
            let buf: Buffer<f32, _> = device.retrieve(10, &buf_base).unwrap();

            for (base, cached) in buf_base.iter().zip(buf.iter()) {
                assert_eq!(base, cached);
            }

            let _x = device.unary_ew(&buf_base, |x| x.exp(), |x| x.exp());
            assert_eq!(buf.len(), buf_base.len());
        }
    }
}
