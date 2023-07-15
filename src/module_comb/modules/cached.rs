use core::{cell::RefCell, marker::PhantomData, panic::Location};
use std::collections::HashMap;

use std::rc::Rc;

use crate::{
    flag::AllocFlag,
    module_comb::{Alloc, Module, PtrConv, Retrieve},
    Shape,
};

// creator struct
#[derive(Debug, PartialEq, Eq, Default)]
pub struct Cached<Mods> {
    pd: PhantomData<Mods>,
}

impl<Mods: Default, D: Alloc> Module<D> for Cached<Mods> {
    type Module = CachedModule<Mods, D>;

    fn new() -> Self::Module {
        CachedModule {
            modules: Default::default(),
            cache: RefCell::new(Cache {
                nodes: Default::default(),
            }),
        }
    }
}

pub struct CachedModule<Mods, D: Alloc> {
    modules: Mods,
    cache: RefCell<Cache<D>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cache<D: Alloc> {
    pub nodes: HashMap<&'static Location<'static>, Rc<D::Data<u8, ()>>>,
}

impl<SD: Alloc> Cache<SD> {
    #[track_caller]
    #[inline]
    pub fn get<T, S: Shape, D: Alloc>(
        &mut self,
        device: &D,
        len: usize,
        callback: fn(),
    ) -> D::Data<T, S>
    where
        SD: PtrConv<D>,
        D: PtrConv<SD>,
    {
        let maybe_allocated = self.nodes.get(&Location::caller());
        match maybe_allocated {
            Some(data) => unsafe { SD::convert(&data, AllocFlag::Wrapper) },
            None => self.add_node(device, len, callback),
        }
    }

    #[track_caller]
    pub fn add_node<T, S: Shape, D: Alloc>(
        &mut self,
        device: &D,
        len: usize,
        callback: fn(),
    ) -> D::Data<T, S>
    where
        D: PtrConv<SD>,
    {
        let data = device.alloc::<T, S>(len, AllocFlag::Wrapper);

        let untyped_ptr = unsafe { D::convert(&data, AllocFlag::None) };
        self.nodes.insert(Location::caller(), Rc::new(untyped_ptr));

        callback();

        data
    }
}

impl<Mods, D: Alloc + PtrConv<SimpleDevice>, SimpleDevice: Alloc + PtrConv<D>> Retrieve<D>
    for CachedModule<Mods, SimpleDevice>
{
    #[inline]
    fn retrieve<T, S: crate::Shape>(&self, device: &D, len: usize) -> D::Data<T, S> {
        self.cache.borrow_mut().get(device, len, || {})
    }
}
