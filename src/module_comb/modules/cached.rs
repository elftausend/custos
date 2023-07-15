use core::{panic::Location, mem::{align_of, size_of}};
use std::collections::HashMap;

use std::rc::Rc;

use crate::{module_comb::{Alloc, Retrieve, Module, CPU}, Shape, cpu::CPUPtr, flag::AllocFlag};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Cache<D: Alloc> {
    pub nodes: HashMap<&'static Location<'static>, Rc<D::Data<u8, ()>>>,

}

pub trait Dev {
    type Device: Alloc;
}

#[derive(Debug, PartialEq, Eq, Default)]
pub struct Cached<Mods> {
    modules: Mods,
    // cache: Cache<D>,
}

impl<D, Mods> Retrieve<D> for Cached<Mods> {
    fn retrieve<T, S: crate::Shape>(&self, device: &D, len: usize) -> <D>::Data<T, S>
    where
        D: Alloc 
    {
        device.alloc(len, crate::flag::AllocFlag::None)
    }
}

impl<Mods: Default, D: Alloc> Module<D> for Cached<Mods> {
    type Module = CachedModule<Mods, D>;

    fn new() -> Self::Module {
        CachedModule {
            modules: Default::default(),
            cache: Cache {
                nodes: Default::default(),
            },
        }
    }
}


pub struct CachedModule<Mods, D: Alloc> {
    modules: Mods,
    cache: Cache<D>,
}

pub trait BoundTo<D: Alloc>: Alloc {}
// impl<D: IsCPU, D2: IsCPU> BoundTo<D> for D2 {}

pub trait PtrConv<D: Alloc>: Alloc {
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(data: &Self::Data<T, IS>, flag: AllocFlag) -> D::Data<Conv, OS>;
}

impl<Mods, OtherMods> PtrConv<CPU<OtherMods>> for CPU<Mods> {
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(data: &CPUPtr<T>, flag: AllocFlag) -> CPUPtr<Conv> {
        CPUPtr {
            ptr: data.ptr as *mut Conv,
            len: data.len,
            flag,
            align: Some(align_of::<T>()),
            size: Some(size_of::<T>()),
        }
    }
}

impl<Mods, D: Alloc, SimpleDevice: Alloc + PtrConv<D>> Retrieve<D> for CachedModule<Mods, SimpleDevice> {
    fn retrieve<T, S: crate::Shape>(&self, device: &D, len: usize) -> D::Data<T, S>
    {
        let res = self.cache.nodes.get(&Location::caller()).unwrap();
        unsafe { SimpleDevice::convert(res, AllocFlag::Wrapper) }
    }
}
