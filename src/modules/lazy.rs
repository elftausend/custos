mod exec_iter;
mod lazy_graph;
#[cfg(feature = "graph")]
mod optimization;
mod ty;
mod wrapper;

pub use ty::*;

use crate::{
    op_hint::OpHint, register_buf_copyable, unregister_buf_copyable, AddLayer, AddOperation, Alloc,
    AnyOp, BoxedShallowCopy, Buffer, CachedBuffers, Cursor, Device, ExecNow, HasId, HasModules, Id,
    IsShapeIndep, Module, NoHasher, OnDropBuffer, OnNewBuffer, Parents, ReplaceBuf, Retrieve,
    RunModule, SetOpHint, Setup, ShallowCopy, Shape, UniqueId, Unit, UseGpuOrCpu,
};

#[cfg(feature = "graph")]
use crate::DeviceError;

use core::{
    any::{Any, TypeId},
    cell::{Cell, RefCell},
    fmt::Debug,
    hash::BuildHasherDefault,
    marker::PhantomData,
};
use std::collections::HashSet;

use self::wrapper::LazyWrapper;
pub use lazy_graph::*;

type Buffers = crate::Buffers<Box<dyn BoxedShallowCopy>>;
type AllocatedIds = HashSet<UniqueId, BuildHasherDefault<NoHasher>>;

#[derive(Default)]
pub struct Lazy<'a, Mods, T = f32> {
    pub modules: Mods,
    alloc_later: RefCell<Vec<(Id, fn(&mut Buffers, &mut AllocatedIds, Id, &dyn Any))>>, // could use D generic instead of dyn Any (required LazyModule structure)
    pub buffers: RefCell<Buffers>,
    replaced_buffers: RefCell<Buffers>,
    // `buffers` shares buffers, that are either lazily allocated or instantly (via new, from..).
    // This ensures to only allocate a buffer once, without having to remove the ID/address collision check
    // TODO: remove this, fix id and address collision - then just use `buffers` for duplicate calls
    allocated_ids: RefCell<AllocatedIds>,
    pub graph: RefCell<LazyGraph<Box<dyn BoxedShallowCopy>, T>>,
    cursor: Cell<usize>,
    enabled: Cell<bool>,
    pd: PhantomData<Cell<&'a ()>>,
}

impl<Mods: Debug, T> Debug for Lazy<'_, Mods, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Lazy").field("mods", &self.modules).finish()
    }
}

pub trait LazySetup {
    #[inline]
    fn lazy_setup(&mut self) -> crate::Result<()> {
        Ok(())
    }
}

pub trait LazyRun {
    #[inline]
    fn run(&self) -> crate::Result<()> {
        Ok(())
    }
}

impl<'a, T, Mods: Module<'a, D>, D: LazySetup + Device + 'a> Module<'a, D> for Lazy<'a, Mods, T> {
    type Module = Lazy<'a, Mods::Module, T>;
    // type Data<T, S: Shape> = LazyWrapper<Mods::Data<T, S>>;

    #[inline]
    fn new() -> Self::Module {
        Lazy {
            modules: Mods::new(),
            buffers: Default::default(),
            replaced_buffers: Default::default(),
            graph: Default::default(),
            alloc_later: Default::default(),
            allocated_ids: Default::default(),
            cursor: Default::default(),
            enabled: Cell::new(true),
            pd: Default::default(),
        }
    }
}

impl<T, Mods: AddOperation> AddOperation for Lazy<'_, Mods, T> {
    fn add_op<Args: Parents<N> + crate::AnyOp, const N: usize>(
        &self,
        args: Args,
        op: impl for<'a> Fn(Args::Replicated<'a>) -> crate::Result<()> + 'static,
    ) -> crate::Result<()> {
        if self.enabled.get() {
            self.graph.try_borrow_mut()
            .expect("already borrowed: BorrowMutError - is the inner operation trying to add an operation as well?")
            .add_operation(args, op);
            Ok(())
        } else {
            self.modules.add_op(args, op)
        }
    }

    #[inline]
    fn ops_count(&self) -> usize {
        self.graph.borrow().ops_count()
    }

    #[inline]
    fn set_lazy_enabled(&self, enabled: bool) {
        self.enabled.set(enabled);
    }

    #[inline]
    fn is_lazy_enabled(&self) -> bool {
        self.enabled.get()
    }
}

impl<T, Mods> SetOpHint<T> for Lazy<'_, Mods, T> {
    #[inline]
    fn set_op_hint(&self, op_hint: OpHint<T>) {
        if let Some(op) = self.graph.borrow_mut().operations.last_mut() {
            op.op_hint = op_hint;
        }
    }
}

impl<T, D: Device + 'static, Mods> ExecNow<D> for Lazy<'_, Mods, T> {
    #[inline]
    fn exec_now(
        &self,
        device: &D,
        range_bounds: impl core::ops::RangeBounds<usize>,
    ) -> crate::Result<()> {
        self.alloc_later(device);
        unsafe {
            self.graph.borrow_mut().call_range::<D>(
                device,
                range_bounds,
                &mut self.buffers.borrow_mut(),
            )?;
        }
        Ok(())
    }
}

impl<T, Mods> Lazy<'_, Mods, T> {
    #[inline]
    pub fn call_lazily<D: Device + 'static>(&self, device: &D) -> crate::Result<()> {
        self.graph
            .borrow_mut()
            .call_lazily(device, &mut self.buffers.borrow_mut())?;
        Ok(())
    }

    pub fn alloc_later<D: 'static>(&self, device: &D) {
        let mut buffers = self.buffers.borrow_mut();
        let mut allocated_ids = self.allocated_ids.borrow_mut();
        for (id, alloc_fn) in self.alloc_later.borrow_mut().drain(..) {
            alloc_fn(&mut buffers, &mut allocated_ids, id, device);
        }
    }
}

impl<T, D: LazySetup, Mods: Setup<D>> Setup<D> for Lazy<'_, Mods, T> {
    #[inline]
    fn setup(device: &mut D) -> crate::Result<()> {
        device.lazy_setup()?;
        Mods::setup(device)
    }
}

impl<T, Mods: RunModule<D>, D: LazyRun + Device + 'static> RunModule<D> for Lazy<'_, Mods, T> {
    #[inline]
    fn run(&self, device: &D) -> crate::Result<()> {
        self.alloc_later(device);
        self.call_lazily::<D>(device)?;
        device.run()?;
        self.modules.run(device)
    }
}

impl<T2, Mods: OnDropBuffer> OnDropBuffer for Lazy<'_, Mods, T2> {
    #[inline]
    fn on_drop_buffer<T: crate::Unit, D: Device, S: Shape>(
        &self,
        device: &D,
        buf: &Buffer<T, D, S>,
    ) {
        unregister_buf_copyable(&mut self.buffers.borrow_mut(), buf.id());
        self.modules.on_drop_buffer(device, buf)
    }
}

impl<'a, T, D, Mods, S, T2> OnNewBuffer<'a, T, D, S> for Lazy<'_, Mods, T2>
where
    T: Unit + 'static,
    D: Device + IsShapeIndep + 'static,
    D::Data<T, S>: ShallowCopy,
    Mods: OnNewBuffer<'a, T, D, S>,
    S: Shape,
{
    #[inline]
    unsafe fn on_new_buffer<'s>(&'s self, device: &'a D, new_buf: &'s Buffer<'a, T, D, S>) {
        unsafe { register_buf_copyable(&mut self.buffers.borrow_mut(), new_buf) };
        self.modules.on_new_buffer(device, new_buf)
    }
}

// pass_down_tape_actions!(Lazy);
#[cfg(feature = "autograd")]
impl<Mods: crate::HasAutograd, T> crate::HasAutograd for Lazy<'_, Mods, T> {}

#[cfg(feature = "autograd")]
impl<Mods: crate::GradActions, U> crate::GradActions for Lazy<'_, Mods, U> {
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

impl<T, Mods: crate::AddGradFn> crate::AddGradFn for Lazy<'_, Mods, T> {
    #[inline]
    fn add_grad_fn<Args: Parents<N> + AnyOp, const N: usize>(
        &self,
        args: Args,
        op: impl for<'b> Fn(Args::Replicated<'b>) -> crate::Result<()> + 'static,
    ) {
        self.modules.add_grad_fn(args, op)
    }

    #[inline]
    fn backward(&mut self) {
        self.modules.backward()
    }

    #[inline]
    fn set_grad_enabled(&self, enabled: bool) {
        self.modules.set_grad_enabled(enabled)
    }
}
// pass_down_grad_fn!(Lazy);
// impl_remove_layer!(Lazy);
impl<Mods, T> crate::RemoveLayer<Mods> for Lazy<'_, Mods, T> {
    #[inline]
    fn inner_mods(self) -> Mods {
        self.modules
    }
}
impl<'a, T, NewMods, SD> AddLayer<NewMods, SD> for Lazy<'a, (), T> {
    type Wrapped = crate::Lazy<'a, NewMods, T>;

    #[inline]
    fn wrap_layer(inner_mods: NewMods) -> Self::Wrapped {
        Lazy {
            modules: inner_mods,
            buffers: Default::default(),
            replaced_buffers: Default::default(),
            graph: Default::default(),
            alloc_later: Default::default(),
            allocated_ids: Default::default(),
            cursor: Default::default(),
            enabled: Cell::new(true),
            pd: Default::default(),
        }
    }
}

impl<T, Mods, D, S, T2> Retrieve<D, T, S> for Lazy<'_, Mods, T2>
where
    T: Unit + 'static,
    Mods: Retrieve<D, T, S>,
    D: IsShapeIndep + 'static,
    D::Data<T, S>: ShallowCopy,
    S: Shape,
{
    #[inline]
    unsafe fn retrieve<const NUM_PARENTS: usize>(
        &self,
        _device: &D,
        len: usize,
        _parents: impl Parents<NUM_PARENTS>,
    ) -> crate::Result<Self::Wrap<T, D::Base<T, S>>>
    where
        S: Shape,
        D: Alloc<T>,
    {
        let mut alloc_later = self.alloc_later.borrow_mut();
        let id = Id {
            id: self.cursor.get() as UniqueId,
            len,
        };

        // alloc later callback in order to keep type information
        alloc_later.push((id, |buffers, allocated_ids, id, device| {
            let device = device.downcast_ref::<D>().unwrap();

            // TODO: remove later if 'buffers' is used for collision avoidance
            if allocated_ids.contains(&id.id) {
                return;
            }

            // TODO: should be fixable - (lazy) -> either return error or fix
            // creating buffers (with data) is not lazy - they are allocated instantly
            // these are then added to `buffers` with their ID (which is the pointing address)
            // new IDs start at 0. 1, 2, 3, ... till a collision with an address happens.
            assert!(
                !buffers.contains_key(&id.id),
                "IDs collided! Maybe pointing address already occupied this ID."
            );

            // safety: AllocFlag::Lazy prevents accessing device when dropping
            let base = device
                .alloc::<S>(id.len, crate::flag::AllocFlag::Lazy)
                .unwrap();
            let data = device.base_to_data(base);
            let buffer = Buffer {
                data,
                device: Some(device),
            };

            let buffer: Buffer<'static, T, D, S> = unsafe { core::mem::transmute(buffer) };
            allocated_ids.insert(id.id);
            buffers.insert(id.id, Box::new(buffer));
        }));

        unsafe { self.bump_cursor() };

        Ok(LazyWrapper {
            data: None,
            id: Some(id),
            _pd: core::marker::PhantomData,
        })
    }

    #[inline]
    fn on_retrieve_finish(&self, retrieved_buf: &Buffer<T, D, S>)
    where
        D: Alloc<T>,
    {
        // unsafe { register_buf(&mut self.buffers.borrow_mut(), retrieved_buf) };

        // pass down
        self.modules.on_retrieve_finish(retrieved_buf)
    }
}

impl<T, Mods> Cursor for Lazy<'_, Mods, T> {
    #[inline]
    fn cursor(&self) -> usize {
        self.cursor.get()
    }

    #[inline]
    unsafe fn set_cursor(&self, cursor: usize) {
        self.cursor.set(cursor)
    }
}

impl<T: Unit + 'static, D: Device + 'static, S: Shape, Mods: OnDropBuffer, T2> ReplaceBuf<T, D, S>
    for Lazy<'_, Mods, T2>
{
    #[inline]
    fn replace_buf<'a, 'b, 'c>(
        &'c self,
        buffer: &'c Buffer<'a, T, D, S>,
    ) -> &'c Buffer<'a, T, D, S> {
        match self.buffers.borrow().get(&buffer.id()) {
            Some(buf) => {
                let mut replaced_buffers = self.replaced_buffers.borrow_mut();
                replaced_buffers.insert(*buffer.id(), buf.shallow_copy());

                let buf = replaced_buffers.get(&buffer.id()).unwrap();
                let buf = &**buf;
                assert_eq!(
                    buf.as_any().type_id(),
                    TypeId::of::<Buffer<T, D, S>>(),
                    "Type data does not match! e.g. optimized graph with different types"
                );
                // extending lifetime is fine -> replaced_buffers is only used for shared references
                unsafe { &*(buf as *const _ as *const Buffer<T, D, S>) }
            }
            None => buffer,
        }
    }
}

impl<T, Mods: UseGpuOrCpu> UseGpuOrCpu for Lazy<'_, Mods, T> {
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

#[cfg(feature = "graph")]
impl<T: crate::Numeric + crate::CDatatype, Mods> crate::Optimize for Lazy<'_, Mods, T> {
    #[inline]
    fn optimize_mem_graph<D: 'static>(
        &self,
        device: &D,
        graph_translator: Option<&crate::GraphTranslator>,
    ) -> crate::Result<()> {
        self.alloc_later_optimized(
            device,
            graph_translator.ok_or(DeviceError::MissingCacheTraces)?,
        )?;
        Ok(())
    }

    #[inline]
    fn unary_fusing<D: crate::UnaryFusing + 'static>(
        &self,
        device: &D,
        graph_translator: Option<&crate::GraphTranslator>,
    ) -> crate::Result<()> {
        self.fuse_unary_ops(
            device,
            graph_translator.ok_or(DeviceError::MissingCacheTraces)?,
        )?;
        Ok(())
    }
}

impl<T, Mods> CachedBuffers for Lazy<'_, Mods, T> {
    #[inline]
    unsafe fn buffers_mut(
        &self,
    ) -> Option<core::cell::RefMut<crate::Buffers<Box<dyn crate::BoxedShallowCopy>>>> {
        Some(self.buffers.borrow_mut())
    }
}

impl<Mods> HasModules for Lazy<'_, Mods> {
    type Mods = Mods;

    #[inline]
    fn modules(&self) -> &Self::Mods {
        &self.modules
    }
}

#[cfg(test)]
mod tests {
    use core::ops::{Add, Deref};

    use crate::{
        tests_helper::{add_ew_slice, AddEw},
        AddOperation, ApplyFunction, Base, Buffer, Combiner, Device, Retrieve, Retriever, Shape,
        Unit, CPU,
    };

    use super::Lazy;

    #[test]
    #[cfg(feature = "cpu")]
    fn test_lazy_retrieve() {
        let device = CPU::<Lazy<Base, i32>>::new();
        let buf = Buffer::<i32, _>::new(&device, 10);
        let res = &buf.data;
        assert_eq!(res.id, None);

        let x: Buffer<i32, _> = device.retrieve(10, ()).unwrap();
        let res = &x.data;
        assert_eq!(res.id, Some(crate::Id { id: 0, len: 10 }));

        let x: Buffer<i32, _> = device.retrieve(10, ()).unwrap();
        let res = &x.data;
        assert_eq!(res.id, Some(crate::Id { id: 1, len: 10 }));
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn test_lazy_apply_fn() {
        let device = CPU::<Lazy<Base, i32>>::new();

        let buf = Buffer::<i32, _>::new(&device, 10);
        let out = device.apply_fn(&buf, |x| x.add(3));

        // assert_eq!(out.read(), &[0; 10]); -- should not work
        device.modules.alloc_later(&device);
        device
            .modules
            .call_lazily::<CPU<Lazy<Base, i32>>>(&device)
            .unwrap();
        // assert_eq!(out.read(), &[3; 10]); -- should work
        assert_eq!(out.replace().read(), &[3; 10]);
        drop(buf);
    }

    #[cfg(feature = "cpu")]
    impl<T, D, S, Mods> AddEw<T, D, S> for CPU<Mods>
    where
        T: Unit + Add<Output = T> + Copy + 'static,
        D: Device + 'static,
        D::Base<T, S>: Deref<Target = [T]>,
        S: Shape,
        Mods: AddOperation + Retrieve<Self, T, S> + 'static,
    {
        #[inline]
        fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S> {
            let mut out = self.retrieve(lhs.len(), ()).unwrap();
            self.add_op((lhs, rhs, &mut out), |(lhs, rhs, out)| {
                add_ew_slice(lhs, rhs, out.as_mut_slice());
                Ok(())
            })
            .unwrap();
            out
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_custom_operation() {
        let device = CPU::<Lazy<Base, i32>>::new();
        let buf = Buffer::<i32, _>::from((&device, &[1, 2, 3, 4, 5, 6, 7, 8]));
        assert_eq!(buf.read(), [1, 2, 3, 4, 5, 6, 7, 8])
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn test_lazy_apply_fn_with_run_cpu_drop_buf() {
        use crate::{DeviceError, Run};

        let device = CPU::<Lazy<Base, i32>>::new();

        {
            let buf = Buffer::<i32, _>::new(&device, 10);
            let _out = device.apply_fn(&buf, |x| x.add(3));
            // assert_eq!(out.replace().read(), &[0; 10]);
        }

        if DeviceError::InvalidLazyBuf != *device.run().err().unwrap().downcast().unwrap() {
            panic!("")
        }
    }
    #[test]
    #[cfg(feature = "cpu")]
    fn test_lazy_apply_fn_with_run_cpu() {
        use crate::Run;

        let device = CPU::<Lazy<Base, i32>>::new();

        let buf = Buffer::<i32, _>::new(&device, 10);
        let out = device.apply_fn(&buf, |x| x.add(3));

        // assert_eq!(out.read(), &[0; 10]);
        device.run().unwrap();
        assert_eq!(out.replace().read(), &[3; 10]);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_lazy_alloc_later() {
        use crate::Run;

        let device = CPU::<Lazy<Base, i32>>::new();

        let buf = Buffer::<i32, _>::new(&device, 10);
        let out = device.apply_fn(&buf, |x| x.add(3));

        device.modules.alloc_later(&device);
        assert_eq!(out.replace().read(), &[0; 10]);
        device.run().unwrap();
        assert_eq!(out.replace().read(), &[3; 10]);
    }

    #[test]
    #[cfg(feature = "opencl")]
    fn test_lazy_apply_fn_with_run_cl() {
        use crate::{ApplyFunction, OpenCL, Run};

        let device = OpenCL::<Lazy<Base, i32>>::new(0).unwrap();

        let buf = Buffer::<i32, _>::new(&device, 10);
        let out = device.apply_fn(&buf, |x| x.add(3));

        device.run().unwrap();
        assert_eq!(out.replace().read(), &[3; 10]);
    }
    #[test]
    #[cfg(feature = "cpu")]
    fn test_lazy_add_apply_fn_with_run() {
        use crate::Run;

        let device = CPU::<Lazy<Base, i32>>::new();

        let buf = Buffer::<i32, _>::new(&device, 10);
        let lhs = device.apply_fn(&buf, |x| x.add(3));

        // assert_eq!(lhs.read(), &[0; 10]);
        let rhs = Buffer::<_, _>::from((&device, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));

        assert_eq!(rhs.read(), &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let out = device.add(&lhs, &rhs);
        // assert_eq!(out.read(), &[0; 10]);

        device.run().unwrap();
        assert_eq!(lhs.replace().read(), &[3; 10]);

        assert_eq!(out.replace().read(), [4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    }

    #[test]
    // #[should_panic]
    #[cfg(feature = "cpu")]
    fn test_lazy_loop_add_apply_fn_with_run() {
        use crate::UnaryGrad;

        let device = CPU::<Lazy<Base>>::new();

        let lhs = Buffer::<i32, _>::new(&device, 10);
        let mut lhs_grad = lhs.empty_like();
        let out_grad = device.buffer([1; 10]);

        for _ in 0..100 {
            device.add_unary_grad(&lhs, &mut lhs_grad, &out_grad, |x| x.add(1));
        }

        // assert_eq!(lhs_grad.as_slice(), [0; 10]);

        // unsafe { device.run().unwrap() };

        // assert_eq!(lhs_grad.as_slice(), [100; 10]);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_lazy_exec_with_range() {
        use crate::{ExecNow, Run};

        let device = CPU::<Lazy<Base>>::new();
        let mut out: Buffer<i32, _, ()> = device.retrieve(4, ()).unwrap();

        device
            .add_op(&mut out, |out| {
                out.clear();
                Ok(())
            })
            .unwrap();

        {
            let a = Buffer::<i32, _, ()>::from_slice(&device, &[1, 2, 3, 4]);
            let b = Buffer::<i32, _, ()>::from_slice(&device, &[1, 2, 3, 4]);
            device
                .add_op((&a, &b, &mut out), |(a, b, out)| {
                    for ((lhs, rhs), out) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
                        *out = lhs + rhs;
                    }
                    Ok(())
                })
                .unwrap();
            device.exec_now(&device, 1..).unwrap();
            assert_eq!(out.replace().as_slice(), [2, 4, 6, 8])
        }
        device.run().unwrap();
        assert_eq!(out.replace().as_slice(), [0; 4])
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_lazy_exec_last_n() {
        use crate::{ExecNow, Run};

        let device = CPU::<Lazy<Base>>::new();
        let mut out: Buffer<i32, _, ()> = device.retrieve(4, ()).unwrap();

        device
            .add_op(&mut out, |out| {
                out.clear();
                Ok(())
            })
            .unwrap();

        {
            let a = Buffer::<i32, _, ()>::from_slice(&device, &[1, 2, 3, 4]);
            let b = Buffer::<i32, _, ()>::from_slice(&device, &[1, 2, 3, 4]);
            device
                .add_op((&a, &b, &mut out), |(a, b, out)| {
                    for ((lhs, rhs), out) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
                        *out = lhs + rhs;
                    }
                    Ok(())
                })
                .unwrap();
            device.exec_last_n(&device, 1).unwrap();
            assert_eq!(out.replace().as_slice(), [2, 4, 6, 8])
        }
        device.run().unwrap();

        assert_eq!(out.replace().as_slice(), [0; 4])
    }

    #[cfg(feature = "cpu")]
    // #[ignore = "causes UB"]
    #[test]
    fn test_lazy_exec_ub_testing() {
        use crate::Run;

        let device = CPU::<Lazy<Base>>::new();

        let mut out: Buffer<i32, _> = device.retrieve(4, ()).unwrap();

        device
            .add_op(&mut out, |out| {
                out.clear();
                Ok(())
            })
            .unwrap();

        {
            let a = Buffer::<i32, _, ()>::from_slice(&device, &[1, 2, 3, 4]);
            let a = a.to_deviceless();
            let b = Buffer::<i32, _, ()>::from_slice(&device, &[1, 2, 3, 4]);
            device
                .add_op((&mut out, &b), move |(out, b)| {
                    for ((lhs, rhs), out) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
                        *out = lhs + rhs;
                    }
                    Ok(())
                })
                .unwrap();
        }

        if device.run().is_ok() {
            panic!()
        }
    }

    // #[cfg(feature = "cpu")]
    // #[test]
    // fn test_owned_buf_in_add_op_should_comp_fail() {
    //     let device = CPU::<Lazy<Base>>::new();
    //     let mut buf = device.buffer([1, 2, 3, 4]);
    //     device.add_op(buf, |buf| {
    //         Ok(())
    //     });

    //     device.add_op(buf.no_id(), |buf| {
    //         Ok(())
    //     });
    // }

    #[cfg(feature = "cached")]
    #[cfg(feature = "cpu")]
    #[test]
    fn test_lazy_cached_two_producers() {
        use crate::Cached;

        let device = CPU::<Lazy<Cached<Base>>>::new();

        let lhs = device.buffer([1, 2, 3, 4]);
        let rhs = device.buffer([1, 2, 3, 4]);

        let _out: Buffer<i32, _> = device.retrieve(10, (&lhs, &rhs)).unwrap();
    }

    /*
    #[cfg(feature = "cpu")]
    #[should_panic]
    #[ignore = "currently wrong panic reasion"]
    #[test]
    fn test_lazy_exec_ub_testing_semi_fixed() {
        use crate::{HasId, Run};

        let device = CPU::<Lazy<Base>>::new();

        let mut out: Buffer<i32, _> = device.retrieve(4, ());

        {
            let a = Buffer::<i32, _, ()>::from_slice(&device, &[1, 2, 3, 4]);
            let b = Buffer::<i32, _, ()>::from_slice(&device, &[1, 2, 3, 4]);
            let (a, b) = (a.id(), b.id());
            device
                .add_op(&mut out, |out| {
                    let buffers = out.device().modules.buffers.borrow();
                    let a = buffers
                        .get(&a)
                        .expect("a went out of scope")
                        .downcast_ref::<Buffer<i32, CPU<Lazy<Base>>>>()
                        .unwrap();
                    let b = buffers
                        .get(&b)
                        .expect("b went out of scope")
                        .downcast_ref::<Buffer<i32, CPU<Lazy<Base>>>>()
                        .unwrap();
                    for ((lhs, rhs), out) in a.iter().zip(b).zip(out.iter_mut()) {
                        *out = lhs + rhs;
                    }
                    Ok(())
                })
                .unwrap();
        }
        unsafe { device.run().unwrap() };
    }
    */
}
