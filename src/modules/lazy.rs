mod lazy_graph;
mod ty;
pub use ty::*;

use core::{any::Any, cell::RefCell, fmt::Debug, hash::BuildHasherDefault};
use std::collections::HashMap;

use crate::{
    AddOperation, Alloc, Buffer, Device, HasId, Id, Module, NoHasher, OnDropBuffer, OnNewBuffer,
    Parents, PtrConv, Retrieve, Run, Setup, Shape, UniqueId,
};

use self::lazy_graph::LazyGraph;

use super::register_buf;

#[derive(Default)]
pub struct Lazy<Mods> {
    pub modules: Mods,
    outs: RefCell<HashMap<UniqueId, Box<dyn Any>, BuildHasherDefault<NoHasher>>>,
    out_ids: RefCell<Vec<Id>>,
    graph: LazyGraph,
}

impl<Mods: Debug> Debug for Lazy<Mods> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Lazy")
            .field("mods", &self.modules)
            .finish()
    }
}

pub trait LazySetup {
    fn lazy_setup(&mut self) -> crate::Result<()> {
        Ok(())
    }
}

pub trait LazyRun {
    fn run(&self) -> crate::Result<()>;
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

impl<T: Graphable, D: Device, Mods> AddOperation<T, D> for Lazy<Mods> {
    #[inline]
    unsafe fn add_operation<S: Shape>(
        &self,
        out: &mut Buffer<T, D, S>,
        operation: impl Fn(&mut Buffer<T, D, S>),
    ) {
    }

    #[inline]
    fn call_lazily(&self) {
        // for (op, out_id) in self.ops.borrow().iter().zip(self.out_ids.borrow().iter()) {
        //     let mut outs = self.outs.borrow_mut();
        //     let out = &mut **outs.get_mut(out_id).unwrap();
        //     op(out)
        // }
    }
}

impl<D: LazySetup, Mods: Setup<D>> Setup<D> for Lazy<Mods> {
    #[inline]
    fn setup(device: &mut D) -> crate::Result<()> {
        device.lazy_setup()?;
        Mods::setup(device)
    }
}

impl<Mods: Run<D>, D: LazyRun> Run<D> for Lazy<Mods> {
    #[inline]
    fn run(&self, device: &mut D) -> crate::Result<()> {
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
        unsafe { super::register_buf(&mut self.outs.borrow_mut(), new_buf) };
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
    use crate::{AddOperation, Base, Buffer, Combiner, CPU};

    use super::Lazy;

    #[test]
    fn test_lazy_device_use() {
        // let device = CPU::<Lazy<Base>>::new();
        // let data = device.alloc::<f32, ()>(10, crate::flag::AllocFlag::None);
    }

    #[test]
    fn test_lazy_device_use_cuda() {
        // let device = CUDA::<Lazy<Base>>::new();
        // let data = device.alloc::<f32, ()>(10, crate::flag::AllocFlag::None);
    }

    use crate::ApplyFunction;

    #[test]
    #[cfg(feature = "macro")]
    fn test_lazy_execution() {
        let device = CPU::<Lazy<Base>>::new();

        let buf = Buffer::<f32, _>::new(&device, 10);
        let out = device.apply_fn(&buf, |x| x.add(3.));

        device.call_lazily();
        println!("out: {:?}", &*out);

        drop(out);
        drop(buf);
    }
}
