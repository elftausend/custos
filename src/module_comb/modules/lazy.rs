use core::{cell::RefCell, fmt::Debug, marker::PhantomData};

use crate::{
    module_comb::{
        AddOperation, Alloc, Buffer, Device, Module, OnDropBuffer, OnNewBuffer, Parents, Retrieve,
        Setup, TapeActions,
    },
    Shape,
};

#[derive(Default)]
pub struct Lazy<Mods> {
    mods: Mods,
    ops: RefCell<Vec<Box<dyn FnOnce() + 'static>>>,
}

impl<Mods: Debug> Debug for Lazy<Mods> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Lazy")
            .field("mods", &self.mods)
            .field("ops_count", &self.ops.borrow().len())
            .finish()
    }
}

pub trait LazySetup {
    fn lazy_setup(&mut self) {}
}

impl<Mods: Module<D>, D: LazySetup> Module<D> for Lazy<Mods> {
    type Module = Lazy<Mods::Module>;

    #[inline]
    fn new() -> Self::Module {
        Lazy {
            mods: Mods::new(),
            ops: Default::default(),
        }
    }
}

impl<Mods> AddOperation for Lazy<Mods> {
    #[inline]
    fn add_operation(&self, operation: impl FnOnce()) {
        let operation: Box<dyn FnOnce()> = Box::new(operation);
        let operation: Box<dyn FnOnce() + 'static> = unsafe { std::mem::transmute(operation) };
        self.ops.borrow_mut().push(operation)
    }

    #[inline]
    fn call_lazily(&self) {
        for op in self.ops.borrow_mut().drain(..) {
            op()
        }
    }
}

impl<D: LazySetup, Mods: Setup<D>> Setup<D> for Lazy<Mods> {
    #[inline]
    fn setup(device: &mut D) {
        device.lazy_setup();
        Mods::setup(device)
    }
}

impl<Mods: OnDropBuffer> OnDropBuffer for Lazy<Mods> {
    #[inline]
    fn on_drop_buffer<'a, T, D: Device, S: Shape>(&self, device: &'a D, buf: &Buffer<T, D, S>) {
        self.mods.on_drop_buffer(device, buf)
    }
}

impl<T, D: Device, S: Shape, Mods: OnNewBuffer<T, D, S>> OnNewBuffer<T, D, S> for Lazy<Mods> {
    #[inline]
    fn on_new_buffer(&self, device: &D, new_buf: &Buffer<T, D, S>) {
        self.mods.on_new_buffer(device, new_buf)
    }
}

impl<Mods: TapeActions> TapeActions for Lazy<Mods> {
    #[inline]
    fn tape(&self) -> Option<core::cell::Ref<super::Tape>> {
        self.mods.tape()
    }

    #[inline]
    fn tape_mut(&self) -> Option<core::cell::RefMut<super::Tape>> {
        self.mods.tape_mut()
    }
}

impl<Mods: Retrieve<D>, D> Retrieve<D> for Lazy<Mods> {
    #[inline]
    fn retrieve<T, S, const NUM_PARENTS: usize>(
        &self,
        device: &D,
        len: usize,
        parents: impl Parents<NUM_PARENTS>,
    ) -> <D>::Data<T, S>
    where
        T: 'static,
        S: Shape,
        D: crate::module_comb::Alloc,
    {
        self.mods.retrieve(device, len, parents)
    }

    #[inline]
    fn on_retrieve_finish<T, S: Shape>(&self, retrieved_buf: &Buffer<T, D, S>)
    where
        T: 'static,
        D: Device,
    {
        // pass down
        self.mods.on_retrieve_finish(retrieved_buf)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        module_comb::{AddOperation, Alloc, Base, Buffer, CPU},
        Combiner,
    };

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

    use crate::module_comb::ApplyFunction;

    #[test]
    fn test_lazy_execution() {
        let device = CPU::<Base>::new();

        let buf = Buffer::<f32, _>::new(&device, 10);
        let out = device.apply_fn(&buf, |x| x.add(3.));

        device.call_lazily();
        println!("out: {:?}", &*out);

        drop(out);
        drop(buf);
    }
}
