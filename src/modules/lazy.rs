mod lazy_graph;
mod ty;
pub use ty::*;

use crate::{
    AddOperation, Alloc, Buffer, Device, HasId, Id, Module, NoHasher, OnDropBuffer, OnNewBuffer,
    Parents, PtrConv, Retrieve, RunModule, Setup, Shape, UniqueId,
};
use core::{any::Any, cell::RefCell, fmt::Debug, hash::BuildHasherDefault};
use std::collections::HashMap;

use self::lazy_graph::LazyGraph;
use super::register_buf;

#[derive(Default)]
pub struct Lazy<Mods> {
    pub modules: Mods,
    outs: RefCell<HashMap<UniqueId, Box<dyn Any>, BuildHasherDefault<NoHasher>>>,
    out_ids: RefCell<Vec<Id>>,
    graph: RefCell<LazyGraph>,
}

impl<Mods: Debug> Debug for Lazy<Mods> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Lazy").field("mods", &self.modules).finish()
    }
}

pub trait LazySetup {
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

impl<Mods: Module<D>, D: LazySetup> Module<D> for Lazy<Mods> {
    type Module = Lazy<Mods::Module>;

    #[inline]
    fn new() -> Self::Module {
        Lazy {
            modules: Mods::new(),
            outs: Default::default(),
            out_ids: Default::default(),
            graph: Default::default(),
        }
    }
}

impl<T: Graphable, D: Device + PtrConv, Mods: AddOperation<T, D>> AddOperation<T, D>
    for Lazy<Mods>
{
    #[inline]
    fn add_op<S: Shape>(
        &self,
        out: &mut Buffer<T, D, S>,
        operation: impl Fn(&mut Buffer<T, D, S>),
    ) {
        self.out_ids.borrow_mut().push(out.id());
        self.graph.borrow_mut().add_operation(operation)
    }
}

impl<Mods> Lazy<Mods> {
    #[inline]
    pub fn call_lazily<D: Device>(&self) -> crate::Result<()> {
        self.graph
            .borrow_mut()
            .call_lazily::<D>(&self.out_ids.borrow(), &mut self.outs.borrow_mut())?;
        Ok(())
    }
}

impl<D: LazySetup, Mods: Setup<D>> Setup<D> for Lazy<Mods> {
    #[inline]
    fn setup(device: &mut D) -> crate::Result<()> {
        device.lazy_setup()?;
        Mods::setup(device)
    }
}

impl<Mods: RunModule<D>, D: LazyRun + PtrConv> RunModule<D> for Lazy<Mods> {
    #[inline]
    fn run(&self, device: &D) -> crate::Result<()> {
        self.call_lazily::<D>()?;
        device.run()?;
        self.modules.run(device)
    }
}

impl<Mods: OnDropBuffer> OnDropBuffer for Lazy<Mods> {
    #[inline]
    fn on_drop_buffer<'a, T, D: Device, S: Shape>(&self, device: &'a D, buf: &Buffer<T, D, S>) {
        super::unregister_buf(&mut self.outs.borrow_mut(), buf.id());
        self.modules.on_drop_buffer(device, buf)
    }
}

impl<T: 'static, D: Device + PtrConv + 'static, S: Shape, Mods: OnNewBuffer<T, D, S>>
    OnNewBuffer<T, D, S> for Lazy<Mods>
{
    #[inline]
    fn on_new_buffer(&self, device: &D, new_buf: &Buffer<T, D, S>) {
        // unsafe { super::register_buf(&mut self.outs.borrow_mut(), new_buf) };
        self.modules.on_new_buffer(device, new_buf)
    }
}

#[cfg(feature = "autograd")]
impl<Mods: crate::TapeActions> crate::TapeActions for Lazy<Mods> {
    #[inline]
    fn tape(&self) -> Option<core::cell::Ref<super::Tape>> {
        self.modules.tape()
    }

    #[inline]
    fn tape_mut(&self) -> Option<core::cell::RefMut<super::Tape>> {
        self.modules.tape_mut()
    }
}

impl<T: 'static, Mods: Retrieve<D, T>, D: PtrConv + 'static> Retrieve<D, T> for Lazy<Mods> {
    #[inline]
    fn retrieve<S, const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        parents: impl Parents<NUM_PARENTS>,
    ) -> <D>::Data<T, S>
    where
        S: Shape,
        D: Alloc<T>,
    {
        self.modules.retrieve(device, len, parents)
    }

    #[inline]
    fn on_retrieve_finish<S: Shape>(&self, retrieved_buf: &Buffer<T, D, S>)
    where
        D: Alloc<T>,
    {
        unsafe { register_buf(&mut self.outs.borrow_mut(), retrieved_buf) };

        // pass down
        self.modules.on_retrieve_finish(retrieved_buf)
    }
}

#[cfg(test)]
mod tests {
    use core::ops::Add;

    use crate::{
        AddOperation, ApplyFunction, Base, Buffer, Combiner, Device, MainMemory, Retrieve,
        Retriever, Shape, CPU,
    };

    use super::Lazy;

    #[test]
    #[cfg(feature = "cpu")]
    fn test_lazy_apply_fn() {
        let device = CPU::<Lazy<Base>>::new();

        let buf = Buffer::<i32, _>::new(&device, 10);
        let out = device.apply_fn(&buf, |x| x.add(3));

        assert_eq!(out.read(), &[0; 10]);
        device.modules.call_lazily::<CPU<Lazy<Base>>>().unwrap();
        assert_eq!(out.read(), &[3; 10]);

        drop(out);
        drop(buf);
    }

    trait AddEw<T, D: Device, S: Shape>: Device {
        fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S>;
    }

    fn add_ew_slice<T: Add<Output = T> + Copy>(lhs: &[T], rhs: &[T], out: &mut [T]) {
        for ((lhs, rhs), out) in lhs.iter().zip(rhs.iter()).zip(out.iter_mut()) {
            *out = *lhs + *rhs;
        }
    }

    impl<T, D, S, Mods> AddEw<T, D, S> for CPU<Mods>
    where
        T: Add<Output = T> + Copy,
        D: MainMemory,
        S: Shape,
        Mods: AddOperation<T, Self> + Retrieve<Self, T>,
    {
        #[inline]
        fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S> {
            let mut out = self.retrieve(lhs.len(), ());
            self.add_op(&mut out, |out| add_ew_slice(lhs, rhs, out));
            out
        }
    }

    #[test]
    fn test_custom_operation() {
        let device = CPU::<Lazy<Base>>::new();
        let buf = Buffer::<_, _>::from((&device, &[1, 2, 3, 4, 5, 6, 7, 8]));
        assert_eq!(buf.read(), [1, 2, 3, 4, 5, 6, 7, 8])
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn test_lazy_apply_fn_with_run_cpu_drop_buf() {
        use crate::{DeviceError, Run};

        let device = CPU::<Lazy<Base>>::new();

        {
            let buf = Buffer::<i32, _>::new(&device, 10);
            let out = device.apply_fn(&buf, |x| x.add(3));
            assert_eq!(out.read(), &[0; 10]);
        }

        if DeviceError::InvalidLazyOutBuf != *device.run().err().unwrap().downcast().unwrap() {
            panic!("")
        }
    }
    #[test]
    #[cfg(feature = "cpu")]
    fn test_lazy_apply_fn_with_run_cpu() {
        use crate::Run;

        let device = CPU::<Lazy<Base>>::new();

        let buf = Buffer::<i32, _>::new(&device, 10);
        let out = device.apply_fn(&buf, |x| x.add(3));

        assert_eq!(out.read(), &[0; 10]);
        device.run().unwrap();
        assert_eq!(out.read(), &[3; 10]);
    }

    #[test]
    #[cfg(feature = "opencl")]
    fn test_lazy_apply_fn_with_run_cl() {
        use crate::{ApplyFunction, OpenCL, Run};

        let device = OpenCL::<Lazy<Base>>::new(0).unwrap();

        let buf = Buffer::<i32, _>::new(&device, 10);
        let out = device.apply_fn(&buf, |x| x.add(3));

        assert_eq!(out.read(), &[0; 10]);
        device.run().unwrap();
        assert_eq!(out.read(), &[3; 10]);
    }
    #[test]
    #[cfg(feature = "cpu")]
    fn test_lazy_add_apply_fn_with_run() {
        use crate::Run;

        let device = CPU::<Lazy<Base>>::new();

        let buf = Buffer::<i32, _>::new(&device, 10);
        let lhs = device.apply_fn(&buf, |x| x.add(3));

        assert_eq!(lhs.read(), &[0; 10]);
        let rhs = Buffer::<_, _>::from((&device, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));

        assert_eq!(rhs.read(), &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let out = device.add(&lhs, &rhs);
        assert_eq!(out.read(), &[0; 10]);

        device.run().unwrap();
        assert_eq!(lhs.read(), &[3; 10]);

        assert_eq!(out.read(), [4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    }
}
