use core::{any::Any, hash::BuildHasherDefault, mem::transmute};
use std::collections::HashMap;

use crate::{flag::AllocFlag, Alloc, Buffer, Device, Ident, IdentHasher, Shape};

#[derive(Debug, Default)]
pub(crate) struct BorrowingCache {
    pub(crate) cache: HashMap<Ident, Box<dyn Any>, BuildHasherDefault<IdentHasher>>,
}

// TODO: make BorrowedCache unuseable without device (=> Static get methods with D: CacheReturn)
impl BorrowingCache {
    pub(crate) fn add_or_get<'a, T, D, S>(
        &mut self,
        device: &'a D,
        id: Ident,
    ) -> &Buffer<'a, T, D, S>
    where
        T: 'static,
        D: Alloc<'a, T, S> + 'static,
        S: Shape,
    {
        self.add_buf_once(device, id);

        let buf_any = self.cache.get(&id).unwrap();
        buf_any.downcast_ref().unwrap()
    }

    pub(crate) fn add_or_get_mut<'a, T, D, S>(
        &mut self,
        device: &D,
        id: Ident,
    ) -> &mut Buffer<'a, T, D, S>
    where
        T: 'static,
        D: for<'b> Alloc<'b, T, S> + 'static,
        S: Shape,
    {
        self.add_buf_once(device, id);
        let buf_any = self.cache.get_mut(&id).unwrap();
        unsafe { transmute(buf_any.downcast_mut::<Buffer<T, D, S>>().unwrap()) }
    }

    pub(crate) fn add_buf_once<'a, T, D, S>(&mut self, device: &'a D, ident: Ident)
    where
        T: 'static,
        D: Alloc<'a, T, S> + 'static,
        S: Shape,
    {
        if self.cache.get(&ident).is_some() {
            return;
        }

        self.add_buf(device, ident)
    }

    pub(crate) fn add_buf<'a, T, D, S>(&mut self, device: &'a D, ident: Ident)
    where
        T: 'static,
        D: Alloc<'a, T, S> + 'static,
        S: Shape,
    {
        // not using ::new, because this buf would get added to the cache of the device.
        let buf = Buffer {
            ptr: device.alloc(ident.len, AllocFlag::BorrowedCache),
            device: Some(device),
            ident,
        };

        let buf = unsafe { transmute::<_, Buffer<'static, T, D, S>>(buf) };
        self.cache.insert(ident, Box::new(buf));
    }

    #[inline]
    pub(crate) fn get_buf<'a, T, D, S>(&self, id: Ident) -> Option<&Buffer<'a, T, D, S>>
    where
        T: 'static,
        D: Device + 'static,
        S: Shape,
    {
        self.cache.get(&id)?.downcast_ref()
    }

    #[inline]
    pub(crate) fn get_buf_mut<'a, T, D, S>(&mut self, id: Ident) -> Option<&mut Buffer<'a, T, D, S>>
    where
        T: 'static,
        D: Device + 'static,
        S: Shape,
    {
        unsafe { transmute(self.cache.get_mut(&id)?.downcast_mut::<Buffer<T, D, S>>()) }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Ident, CPU};

    use super::BorrowingCache;

    /*#[test]
    fn test_comp_error() {
        let device = CPU::new();


        let a = {
            let mut cache = BorrowingCache::default();
            cache.add_or_get::<f32, CPU, ()>(&device, Ident::new(10))
        };
    }*/

    #[test]
    fn test_get_borrowed() {
        let device = CPU::new();
        let mut cache = BorrowingCache::default();

        let (fid, sid, tid) = (
            Ident::new_bumped(10),
            Ident::new_bumped(10),
            Ident::new_bumped(10),
        );

        cache.add_buf_once::<f32, CPU, ()>(&device, fid);
        cache.add_buf_once::<f32, CPU, ()>(&device, sid);
        cache.add_buf_once::<f32, CPU, ()>(&device, tid);

        let a = cache.get_buf::<f32, CPU, ()>(fid).unwrap();
        let b = cache.get_buf::<f32, CPU, ()>(fid).unwrap();

        assert_eq!(a.ptr, b.ptr);
    }
}
