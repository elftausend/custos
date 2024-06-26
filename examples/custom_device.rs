use core::fmt;
use std::{
    any::Any,
    cell::{Cell, UnsafeCell},
    collections::HashMap,
    hash::BuildHasherDefault,
    marker::PhantomData,
    mem::transmute,
};

use custos::{
    flag::AllocFlag, Alloc, Base, Buffer, CachingError, Device, HasId, Id, NoHasher, OnDropBuffer,
    PtrType, Shape, UniqueId, Unit, WrappedData, CPU,
};

pub trait Module<'a, D: 'a, Mods = ()> {
    type Module;

    fn new() -> Self::Module;
}

pub trait Str {
    fn str(&self) -> &String;
}

pub trait New<SimpleMods> {
    fn new1<'a, NewMods>() -> CPU<SimpleMods::Module>
    where
        Self: 'a,
        SimpleMods: Module<'a, CPU, Module = NewMods>;
}

impl<SimpleMods> New<SimpleMods> for CPU<SimpleMods> {
    #[inline]
    fn new1<'a, NewMods>() -> CPU<SimpleMods::Module>
    where
        Self: 'a,
        SimpleMods: Module<'a, CPU, Module = NewMods>,
    {
        CPU {
            modules: SimpleMods::new(),
        }
    }
}

#[derive(Default)]
pub struct Autograd<'a, Mods> {
    _cache: UnsafeCell<BorrowCache<'a>>,
    val: Cell<Option<&'a f32>>,
    _modules: Mods,
}

impl<'a, Mods> Autograd<'a, Mods> {
    pub fn add_buf<OtherMods>(&'a self, device: &'a CPU<OtherMods>) {
        // unsafe { (*self._cache.get()).add_buf(device) };
        // binding.get_buf_mut(device);
        // self.val.set(Some(&device.val));
    }
}

pub trait AnyBuffer {}

// impl fmt::Debug for dyn AnyBuffer {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         f.debug_struct("Any").finish_non_exhaustive()
//     }
// }
// impl dyn AnyBuffer {
//     pub fn is<T: 'static>(&self) -> bool {
//         TypeId::of::<T>() == self.type_id()
//     }

//     #[inline]
//     pub unsafe fn downcast_mut_unchecked<T: Any>(&mut self) -> &mut T {
//         debug_assert!(self.is::<T>());
//         // SAFETY: caller guarantees that T is the correct type
//         unsafe { &mut *(self as *mut dyn AnyBuffer as *mut T) }
//     }
// }

#[derive(Default)]
pub struct BorrowCache<'a> {
    cache: HashMap<UniqueId, Box<dyn Any>, BuildHasherDefault<NoHasher>>,
    pd: PhantomData<&'a ()>,
}

impl<'dev> BorrowCache<'dev> {
    pub fn add_buf_once<T, D, S>(&mut self, device: &'dev D, id: Id, new_buf: &mut bool)
    where
        T: Unit + 'static,
        D: Alloc<T> + 'static,
        S: Shape,
    {
        if self.cache.contains_key(&id) {
            return;
        }
        *new_buf = true;
        self.add_buf::<T, D, S>(device, id)
    }

    pub fn add_buf<T, D, S>(&mut self, device: &'dev D, id: Id)
    where
        T: Unit + 'static,
        D: Alloc<T> + 'static,
        S: Shape,
    {
        // not using ::new, because this buf would get added to the cache of the device.
        // not anymore ?
        let buf = Buffer {
            data: device.base_to_data(device.alloc::<S>(id.len, AllocFlag::BorrowedCache).unwrap()),
            device: Some(device),
        };

        let buf = unsafe { transmute::<_, Buffer<'static, T, D, S>>(buf) };
        self.cache.insert(*id, Box::new(buf));
    }

    #[inline]
    pub fn get_buf<T, D, S>(&self, id: Id) -> Result<&Buffer<'dev, T, D, S>, CachingError>
    where
        T: Unit + 'static,
        D: Device + 'static,
        S: Shape,
    {
        self.cache
            .get(&id)
            .ok_or(CachingError::InvalidId)?
            .downcast_ref()
            .ok_or(CachingError::InvalidTypeInfo)
    }

    #[inline]
    pub fn get_buf_mut<'a, T, D, S>(
        &'a mut self,
        id: Id,
    ) -> Result<&'a mut Buffer<'dev, T, D, S>, CachingError>
    where
        T: Unit + 'static,
        D: Device + 'static,
        S: Shape,
    {
        unsafe {
            transmute::<Result<&'a mut Buffer<'static, T, D, S>, CachingError>, _>(
                self.cache
                    .get_mut(&id)
                    .ok_or(CachingError::InvalidId)?
                    .downcast_mut::<Buffer<T, D, S>>()
                    .ok_or(CachingError::InvalidTypeInfo),
            )
        }
    }
}

impl<'a, D: 'a, Mods: Module<'a, D>> Module<'a, D> for Autograd<'a, Mods> {
    type Module = Autograd<'a, Mods::Module>;

    fn new() -> Self::Module {
        Autograd {
            _cache: Default::default(),
            _modules: Mods::new(),
            val: Default::default(),
        }
    }
}

impl<'a, Mods: OnDropBuffer> OnDropBuffer for Autograd<'a, Mods> {
    #[inline]
    fn on_drop_buffer<T: Unit, D: Device, S: Shape>(
        &self,
        _device: &D,
        _buf: &custos::prelude::Buffer<T, D, S>,
    ) {
        self._modules.on_drop_buffer(_device, _buf)
    }
}

impl<'a, Mods: WrappedData> WrappedData for Autograd<'a, Mods> {
    type Wrap<T, Base: HasId + PtrType> = Mods::Wrap<T, Base>;

    #[inline]
    fn wrap_in_base<T, Base: HasId + PtrType>(&self, base: Base) -> Self::Wrap<T, Base> {
        self._modules.wrap_in_base(base)
    }

    #[inline]
    fn wrapped_as_base<T, Base: HasId + PtrType>(wrap: &Self::Wrap<T, Base>) -> &Base {
        Mods::wrapped_as_base(wrap)
    }

    #[inline]
    fn wrapped_as_base_mut<T, Base: HasId + PtrType>(wrap: &mut Self::Wrap<T, Base>) -> &mut Base {
        Mods::wrapped_as_base_mut(wrap)
    }
}

impl<'a, D: 'a> Module<'a, D> for Base {
    type Module = Base;

    fn new() -> Self::Module {
        Base
    }
}

fn main() {
    let mut cache = BorrowCache::default();
    let mods = Autograd::<Base>::default();
    {
        let dev = CPU::<Autograd<Base>>::new1();
        cache.add_buf::<i32, _, ()>(&dev, Id { id: 0, len: 10 });
        // dev.modules.add_buf(&dev);
        // let out = dev.modules._cache._cache.get(&3).unwrap();
        // mods.add_buunsafe { f(&dev);
        // mods.add_buf(&dev);
        {
            // cache.add_buf(&dev);
        }
        {
            // cache.add_buf(&dev);
        }
        // cache.get_buf_mut(&dev);
        let out = cache.get_buf::<i32, CPU<Autograd<Base>>, ()>(Id { id: 0, len: 10 }).unwrap();
        let out1 = cache
            .get_buf_mut::<i32, CPU<Autograd<Base>>, ()>(Id { id: 0, len: 10 })
            .unwrap();
        // assert_eq!(out.len(), out1.len());
    }
    // let out = unsafe { cache.get_buf::<i32, CPU<Autograd<Base>>, ()>(Id { id:0, len: 10}) };
    let dev = CPU::<Autograd<Base>>::new1();
    // cache.add_buf(&dev);
    // mods.val;
}
