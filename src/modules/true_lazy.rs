mod lazy_graph;
mod ty;

use crate::{
    flag::AllocFlag, AddOperation, Device, HasId, LazySetup, MainMemory, Module, PtrConv, PtrType,
    Retrieve, Setup, Shape,
};
use core::{
    alloc::Layout,
    cell::{Cell, RefCell},
    mem::{align_of, size_of},
    ptr::null_mut,
};
use lazy_graph::LazyGraph;
use std::alloc::handle_alloc_error;
use std::collections::HashMap;

pub struct TrueLazy<Mods> {
    modules: Mods,
    graph: LazyGraph,
    id_to_ptr: RefCell<Vec<(Layout, *mut *mut u64)>>,
    has_allocated: bool,
    id: Cell<u64>,
}

impl<Mods: Module<D>, D: LazySetup> Module<D> for TrueLazy<Mods> {
    type Module = TrueLazy<Mods::Module>;

    #[inline]
    fn new() -> Self::Module {
        TrueLazy {
            modules: Mods::new(),
            graph: LazyGraph::default(),
            id_to_ptr: Default::default(),
            has_allocated: false,
            id: Cell::default(),
        }
    }
}

impl<Mods> TrueLazy<Mods> {
    pub fn alloc(&mut self) {
        self.has_allocated = true;
        for (layout, ptr) in self.id_to_ptr.borrow_mut().iter_mut() {
            // let allocated_ptr = unsafe { std::alloc::alloc(*layout) };

            // if allocated_ptr.is_null() {
            //     handle_alloc_error(*layout)
            // }

            // unsafe { *ptr = allocated_ptr as u64 as *mut u64 }
        }
    }
}

impl<Mods> Drop for TrueLazy<Mods> {
    fn drop(&mut self) {
        if !self.has_allocated {
            return;
        }
        for (layout, ptr) in self.id_to_ptr.borrow_mut().iter() {
            unsafe { std::alloc::dealloc((**ptr) as *mut u8, *layout) };
        }
    }
}

impl<Mods, D> Setup<D> for TrueLazy<Mods> {}

impl<Mods, T, D, S: Shape> Retrieve<D, T, S> for TrueLazy<Mods>
where
    D: Device,
    D::Data<T, S>: Default
{
    fn retrieve<const NUM_PARENTS: usize>(
        &self,
        _device: &D,
        len: usize,
        parents: impl crate::Parents<NUM_PARENTS>,
        _alloc_fn: impl FnOnce(&D, AllocFlag) -> D::Data<T, S>,
    ) -> <D>::Data<T, S>
    where
        S: Shape,
        D: Device + crate::Alloc<T>,
    {
        let mut data = D::Data::<T, S>::default();
unsafe {        data.set_flag(AllocFlag::Dangling);}
        let id = self.id.take();

        unsafe {
            data.set_id(id);
            data.set_size(len)
        }

        let id_mut = data.id_mut();

            let ptr = unsafe {
                *id_mut
            };

            println!("ptr: {ptr:?}");
        self.id_to_ptr.borrow_mut().push((
            Layout::from_size_align(len * size_of::<T>(), align_of::<T>()).unwrap(),
            id_mut,
        ));

        self.id.set(id + 1);
        data
    }
}

impl<Mods, D: Device + PtrConv> AddOperation<D> for TrueLazy<Mods> {
    fn add_operation2<T, S: Shape>(
        &self,
        out: &mut crate::Buffer<T, D, S>,
        operation: impl Fn(&mut crate::Buffer<T, D, S>),
    ) {
        // self.graph.add_operation(operation)
    }

    fn call_lazily(&self) {}
}

#[cfg(feature = "autograd")]
impl<Mods: crate::TapeActions> crate::TapeActions for TrueLazy<Mods> {
    #[inline]
    fn tape(&self) -> Option<core::cell::Ref<super::Tape>> {
        self.modules.tape()
    }

    #[inline]
    fn tape_mut(&self) -> Option<core::cell::RefMut<super::Tape>> {
        self.modules.tape_mut()
    }
}

#[cfg(test)]
mod tests {
    use core::ptr::null_mut;

    pub struct Ptr {
        ptr: *mut u8,
    }
    #[test]
    fn test_update_pointer_in_struct_afterwards() {
        let mut ptr = Ptr { ptr: null_mut() };
        let ptr_ref = (&mut ptr.ptr) as *mut *mut u8;

        unsafe { *ptr_ref = 3u64 as *mut u64 as *mut u8 }

        println!("ptr: {:?}", ptr.ptr as u64);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_true_lazy_id_set() {
        use core::ptr::null_mut;

        use crate::{Base, Buffer, HasId, Id, TrueLazy, CPU};

        #[allow(unused_assignments)]
        let mut ptr: *mut u8 = null_mut();
        ptr = 3usize as *mut usize as *mut u8;

        assert_eq!(ptr as usize, 3);

        let device = CPU::<TrueLazy<Base>>::new();
        let buf = Buffer::<i32, _>::new(&device, 10);
        assert_eq!(buf.id(), Id { id: 0, len: 10 });
        let buf = Buffer::<i32, _>::new(&device, 10);
        assert_eq!(buf.id(), Id { id: 1, len: 10 });
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_allocating_of_true_lazy_bufs() {
        use crate::{Base, Buffer, HasId, Id, TrueLazy, CPU};

        let mut device = CPU::<TrueLazy<Base>>::new();
        let buf = Buffer::<i32, _>::new(&device, 10);
        assert_eq!(buf.id(), Id { id: 0, len: 10 });
        let buf = Buffer::<i32, _>::new(&device, 10);
        assert_eq!(buf.id(), Id { id: 1, len: 10 });

        device.modules.alloc();
        device.modules.has_allocated = false;
        println!("here")
    }
}
