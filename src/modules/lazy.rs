use core::{any::Any, fmt::Debug};

use crate::{
    flag::AllocFlag, AddOperation, Alloc, Buffer, Device, Module, Operation, Parents, PtrConv,
    Retrieve, Run, Setup, Shape,
};

#[derive(Default)]
pub struct Lazy<Mods> {
    pub modules: Mods,
    // outs: RefCell<HashMap<UniqueId, Box<dyn Any>, BuildHasherDefault<NoHasher>>>,
    // ops: RefCell<Vec<Box<dyn Fn(&mut dyn Any)>>>,
    // out_ids: RefCell<Vec<Id>>,
    // ops2: RefCell<Vec<Box<dyn Operation>>>,
}

impl<Mods: Debug> Debug for Lazy<Mods> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Lazy")
            .field("mods", &self.modules)
            // .field("ops_count", &self.ops.borrow().len())
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
        }
    }
}
/*
impl<Mods> AddOperation for Lazy<Mods> {
    #[inline]
    unsafe fn add_operation<T: 'static, D: Device + 'static, S: Shape>(
        &self,
        out: &mut Buffer<T, D, S>,
        operation: impl Fn(&mut dyn Any),
    ) {
        // // operation(out);
        // self.out_ids.borrow_mut().push(out.id());
        // let operation: Box<dyn Fn(&mut dyn Any)> = Box::new(operation);
        // let operation: Box<dyn Fn(&mut dyn Any) + 'static> =
        //     unsafe { std::mem::transmute(operation) };
        // self.ops.borrow_mut().push(operation)
    }

    #[inline]
    fn add_operation2<T, D: Device, S: Shape>(
        &self,
        out: &mut Buffer<T, D, S>,
        operation: impl Fn(&mut Buffer<T, D, S>),
    ) {
        // let operation: Box<dyn Operation> = Box::new(operation);
        // let operation: Box<dyn Operation + 'static> = unsafe { std::mem::transmute(operation) };
        // self.ops2.borrow_mut().push(operation)
    }

    #[inline]
    fn call_lazily(&self) {
        // for (op, out_id) in self.ops.borrow().iter().zip(self.out_ids.borrow().iter()) {
        //     let mut outs = self.outs.borrow_mut();
        //     let out = &mut **outs.get_mut(out_id).unwrap();
        //     op(out)
        // }
    }
}*/

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

impl<T: 'static, S: Shape, Mods: Retrieve<D, T, S>, D: PtrConv + 'static> Retrieve<D, T, S>
    for Lazy<Mods>
{
    #[inline]
    fn retrieve<const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        parents: impl Parents<NUM_PARENTS>,
        alloc_fn: impl FnOnce(&D, AllocFlag) -> D::Data<T, S>,
    ) -> <D>::Data<T, S>
    where
        S: Shape,
        D: Alloc<T>,
    {
        self.modules.retrieve(device, len, parents, alloc_fn)
    }

    #[inline]
    fn on_retrieve_finish(&self, retrieved_buf: &Buffer<T, D, S>)
    where
        D: Alloc<T>,
    {
        // unsafe { register_buf(&mut self.outs.borrow_mut(), retrieved_buf) };

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
