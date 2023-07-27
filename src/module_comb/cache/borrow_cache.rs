use core::{any::Any, hash::BuildHasherDefault, mem::transmute};
use std::collections::HashMap;

use crate::{
    flag::AllocFlag,
    module_comb::{Alloc, Buffer, Id},
    Shape,
};

use super::NoHasher;

pub type UniqueId = u64;

#[derive(Debug, Default)]
pub struct BorrowCache {
    pub cache: HashMap<UniqueId, Box<dyn Any>, BuildHasherDefault<NoHasher>>,
}

// TODO: make BorrowedCache unuseable without device (=> Static get methods with D: CacheReturn)
impl BorrowCache {
    pub fn add_or_get<'a, T, D, S>(&mut self, device: &'a D, id: Id) -> &Buffer<'a, T, D, S>
    where
        T: 'static,
        D: Alloc + 'static,
        S: Shape,
    {
        self.add_buf_once::<T, D, S>(device, id);

        let buf_any = self.cache.get(&id).unwrap();
        buf_any.downcast_ref().unwrap()
    }

    pub fn add_or_get_mut<'a, T, D, S>(&mut self, device: &D, id: Id) -> &mut Buffer<'a, T, D, S>
    where
        T: 'static,
        D: Alloc + 'static,
        S: Shape,
    {
        self.add_buf_once::<T, D, S>(device, id);
        self.get_buf_mut(id).unwrap()
    }

    pub fn add_buf_once<'a, T, D, S>(&mut self, device: &'a D, id: Id)
    where
        T: 'static,
        D: Alloc + 'static,
        S: Shape,
    {
        if self.cache.get(&id).is_some() {
            return;
        }

        self.add_buf::<T, D, S>(device, id)
    }

    pub fn add_buf<'a, T, D, S>(&mut self, device: &'a D, id: Id)
    where
        T: 'static,
        D: Alloc + 'static,
        S: Shape,
    {
        // not using ::new, because this buf would get added to the cache of the device.
        // not anymore ?
        let buf = Buffer {
            data: device.alloc::<T, S>(id.len, AllocFlag::BorrowedCache),
            device: Some(device),
        };

        let buf = unsafe { transmute::<_, Buffer<'static, T, D, S>>(buf) };
        self.cache.insert(*id, Box::new(buf));
    }

    #[inline]
    pub fn get_buf<'a, T, D, S>(&self, id: Id) -> Option<&Buffer<'a, T, D, S>>
    where
        T: 'static,
        D: Alloc + 'static,
        S: Shape,
    {
        self.cache.get(&id)?.downcast_ref()
    }

    #[inline]
    pub fn get_buf_mut<'a, T, D, S>(&mut self, id: Id) -> Option<&mut Buffer<'a, T, D, S>>
    where
        T: 'static,
        D: Alloc + 'static,
        S: Shape,
    {
        unsafe { transmute(self.cache.get_mut(&id)?.downcast_mut::<Buffer<T, D, S>>()) }
    }
}

#[cfg(test)]
mod tests {
    use crate::CPU;

    use super::BorrowCache;

    /*#[test]
    fn test_comp_error() {
        let device = CPU::new();


        let a = {
            let mut cache = BorrowingCache::default();
            cache.add_or_get::<f32, CPU, ()>(&device, Id::new(10))
        };
    }*/

    /*#[test]
    fn test_get_borrowed() {
        let device = CPU::new();
        let mut cache = BorrowCache::default();

        let (fid, sid, tid) = (
            Id::new_bumped(10),
            Id::new_bumped(10),
            Id::new_bumped(10),
        );

        cache.add_buf_once::<f32, CPU, ()>(&device, fid);
        cache.add_buf_once::<f32, CPU, ()>(&device, sid);
        cache.add_buf_once::<f32, CPU, ()>(&device, tid);

        let a = cache.get_buf::<f32, CPU, ()>(fid).unwrap();
        let b = cache.get_buf::<f32, CPU, ()>(fid).unwrap();

        assert_eq!(a.ptr, b.ptr);
    }*/
}
