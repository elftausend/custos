mod lazy_graph;
mod ty;
pub use ty::*;

use crate::{
    pass_down_tape_actions, AddOperation, Alloc, Buffer, Device, ExecNow, HasId, Module, NoHasher,
    OnDropBuffer, OnNewBuffer, Parents, PtrConv, Retrieve, RunModule, Setup, Shape, UniqueId,
    UpdateArgs,
};
use core::{any::Any, cell::RefCell, fmt::Debug, hash::BuildHasherDefault};
use std::collections::HashMap;

pub use self::lazy_graph::LazyGraph;
use super::register_buf;

#[derive(Default)]
pub struct Lazy<Mods> {
    pub modules: Mods,
    buffers: RefCell<HashMap<UniqueId, Box<dyn Any>, BuildHasherDefault<NoHasher>>>,
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

pub struct LazyWrapper<T> {
    data: T,
}

impl<T: HasId> HasId for LazyWrapper<T> {
    fn id(&self) -> crate::Id {
        todo!()
    }
}

impl<Mods: Module<D>, D: LazySetup + Device> Module<D> for Lazy<Mods> {
    type Module = Lazy<Mods::Module>;
    // type Data<T, S: Shape> = LazyWrapper<Mods::Data<T, S>>;

    #[inline]
    fn new() -> Self::Module {
        Lazy {
            modules: Mods::new(),
            buffers: Default::default(),
            graph: Default::default(),
        }
    }
}

impl<Mods: AddOperation> AddOperation for Lazy<Mods> {
    #[inline]
    fn ops_count(&self) -> usize {
        self.graph.borrow().ops.len()
    }

    #[inline]
    fn add_op<Args: Parents<N> + UpdateArgs, const N: usize>(
        &self,
        args: Args,
        operation: fn(&mut Args) -> crate::Result<()>,
    ) -> crate::Result<()> {
        Ok(self.graph.try_borrow_mut()
            .expect("already borrowed: BorrowMutError - is the inner operation trying to add an operation as well?")
            .add_operation(args, operation))
    }
}

impl<D: Device + 'static, Mods> ExecNow<D> for Lazy<Mods> {
    #[inline]
    fn exec_now(&self, range_bounds: impl core::ops::RangeBounds<usize>) -> crate::Result<()> {
        unsafe {
            self.graph
                .borrow_mut()
                .call_range::<D>(range_bounds, &mut self.buffers.borrow_mut())?;
        }
        Ok(())
    }
}

impl<Mods> Lazy<Mods> {
    #[inline]
    pub unsafe fn call_lazily<D: Device + 'static>(&self) -> crate::Result<()> {
        self.graph
            .borrow_mut()
            .call_lazily::<D>(&mut self.buffers.borrow_mut())?;
        // self.graph
        //     .borrow_mut()
        //     .call_lazily::<D>(&self.out_ids.borrow(), &mut self.buffers.borrow_mut())?;
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

impl<Mods: RunModule<D>, D: LazyRun + PtrConv + 'static> RunModule<D> for Lazy<Mods> {
    #[inline]
    fn run(&self, device: &D) -> crate::Result<()> {
        unsafe { self.call_lazily::<D>()? };
        device.run()?;
        self.modules.run(device)
    }
}

impl<Mods: OnDropBuffer> OnDropBuffer for Lazy<Mods> {
    #[inline]
    fn on_drop_buffer<T, D: Device, S: Shape>(&self, device: &D, buf: &Buffer<T, D, S>) {
        super::unregister_buf(&mut self.buffers.borrow_mut(), buf.id());
        self.modules.on_drop_buffer(device, buf)
    }
}

impl<T: 'static, D: Device + PtrConv + 'static, Mods: OnNewBuffer<T, D>> OnNewBuffer<T, D>
    for Lazy<Mods>
{
    #[inline]
    fn on_new_buffer<S: Shape>(&self, device: &D, new_buf: &Buffer<T, D, S>) {
        unsafe { super::register_buf(&mut self.buffers.borrow_mut(), new_buf) };
        self.modules.on_new_buffer(device, new_buf)
    }
}

pass_down_tape_actions!(Lazy);

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
        unsafe { register_buf(&mut self.buffers.borrow_mut(), retrieved_buf) };

        // pass down
        self.modules.on_retrieve_finish(retrieved_buf)
    }
}

#[cfg(test)]
mod tests {
    use core::ops::{Add, Deref};

    use crate::{
        AddOperation, ApplyFunction, Base, Buffer, Combiner, Device, Retrieve, Retriever, Shape,
        CPU,
    };

    use super::Lazy;

    #[test]
    #[cfg(feature = "cpu")]
    fn test_lazy_apply_fn() {
        let device = CPU::<Lazy<Base>>::new();

        let buf = Buffer::<i32, _>::new(&device, 10);
        let out = device.apply_fn(&buf, |x| x.add(3));

        assert_eq!(out.read(), &[0; 10]);
        unsafe { device.modules.call_lazily::<CPU<Lazy<Base>>>().unwrap() }
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
        T: Add<Output = T> + Copy + 'static,
        D: Device + 'static,
        D::Data<T, S>: Deref<Target = [T]>,
        S: Shape,
        Mods: AddOperation + Retrieve<Self, T> + 'static,
    {
        #[inline]
        fn add(&self, lhs: &Buffer<T, D, S>, rhs: &Buffer<T, D, S>) -> Buffer<T, Self, S> {
            let mut out = self.retrieve(lhs.len(), ());
            self.add_op((lhs, rhs, &mut out), |(lhs, rhs, out)| {
                add_ew_slice(lhs, rhs, out);
                Ok(())
            })
            .unwrap();
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

        if DeviceError::InvalidLazyBuf
            != unsafe { *device.run().err().unwrap().downcast().unwrap() }
        {
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
        unsafe { device.run().unwrap() };
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
        unsafe { device.run().unwrap() }
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

        unsafe { device.run().unwrap() };
        assert_eq!(lhs.read(), &[3; 10]);

        assert_eq!(out.read(), [4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
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
        let mut out: Buffer<i32, _, ()> = device.retrieve(4, ());

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
            device.exec_now(1..).unwrap();
            assert_eq!(out.as_slice(), [2, 4, 6, 8])
        }
        unsafe { device.run().unwrap() };
        assert_eq!(out.as_slice(), [0; 4])
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_lazy_exec_last_n() {
        use crate::{ExecNow, Run};

        let device = CPU::<Lazy<Base>>::new();
        let mut out: Buffer<i32, _, ()> = device.retrieve(4, ());

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
            device.exec_last_n(1).unwrap();
            assert_eq!(out.as_slice(), [2, 4, 6, 8])
        }
        unsafe { device.run().unwrap() };

        assert_eq!(out.as_slice(), [0; 4])
    }

    #[cfg(feature = "cpu")]
    // #[ignore = "causes UB"]
    #[test]
    fn test_lazy_exec_ub_testing() {
        use crate::{AsNoId, Run};

        let device = CPU::<Lazy<Base>>::new();

        let mut out: Buffer<i32, _> = device.retrieve(4, ());

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
            let vec = vec![1, 2, 3];
            device
                .add_op(
                    (&mut out, a.no_id(), &b, vec.no_id()),
                    |(out, a, b, _vec)| {
                        for ((lhs, rhs), out) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
                            *out = lhs + rhs;
                        }
                        Ok(())
                    },
                )
                .unwrap();
        }

        if let Ok(_) = unsafe { device.run() } {
            panic!()
        }
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
