use core::{
    any::Any,
    cell::{Ref, RefCell, RefMut}, mem::transmute, fmt::Display,
};
use std::collections::HashMap;

use crate::{Alloc, Buffer, Device, Ident, Shape, Read, prelude::Number, flag::AllocFlag};

#[inline]
fn downcast_ref_buf<'a, T, D, S>(
    buf_any: &'a Box<dyn Any>,
) -> Option<&'a Buffer<'static, T, D, S>>
where
    T: 'static,
    S: Shape + 'static,
    D: Device + 'static,
{    
    buf_any.downcast_ref::<Buffer<T, D, S>>()
}

#[inline]
fn downcast_mut_buf<'a, T, D, S>(
    buf_any: &'a mut Box<dyn Any>,
) -> Option<&'a mut Buffer<'static, T, D, S>>
where
    T: 'static,
    S: Shape + 'static,
    D: Device + 'static,
{    
    buf_any.downcast_mut::<Buffer<T, D, S>>()
}

#[derive(Debug, Default)]
pub(crate) struct BorrowedCache {
    cache: RefCell<HashMap<Ident, Box<dyn Any>>>,
    cache2: HashMap<Ident, Box<dyn Any>>,
}

// TODO: make BorrowedCache unuseable without device (=> Static get methods with D: CacheReturn)
impl BorrowedCache {
    fn add_buf_mut<'cache, 'dev, T, D, S>(&'cache mut self, device: &'dev D, id: Ident) -> Option<&'cache mut Buffer<'static, T, D, S>>
    where
        //'cache: 'dev,
        T: 'static,
        S: Shape + 'static,
        D: for<'b> Alloc<'b, T, S> + 'static,
    {

        // not using ::new, because this buf would get added to the cache of the device.
        let buf = Buffer {
            ptr: device.alloc(id.len, AllocFlag::BorrowedCache),
            device: Some(device),
            ident: id,
            
        };
        let buf = unsafe { transmute::<_, Buffer<'static, T, D, S>>(buf)};
        self.cache2.insert(id, Box::new(buf));        
        downcast_mut_buf(self.cache2.get_mut(&id).unwrap())
    }

    fn add_buf<'cache, 'dev, T, D, S>(&'cache mut self, device: &'dev D, id: Ident) -> Option<&'cache Buffer<'static, T, D, S>>
    where
        //'cache: 'dev,
        T: 'static,
        S: Shape + 'static,
        D: for<'b> Alloc<'b, T, S> + 'static,
    {

        // not using ::new, because this buf would get added to the cache of the device.
        let buf = Buffer {
            ptr: device.alloc(id.len, AllocFlag::BorrowedCache),
            device: Some(device),
            ident: id,
            
        };
        let buf = unsafe { transmute::<_, Buffer<'static, T, D, S>>(buf)};
        self.cache2.insert(id, Box::new(buf));        
        downcast_ref_buf(self.cache2.get(&id).unwrap())
    }

    #[inline]
    fn downcast_ref_buf<'cache, T, D, S>(
        &'cache self,
        buf_any: &'cache (dyn Any + 'static),
    ) -> Option<&'cache Buffer<'static, T, D, S>>
    where
        T: 'static,
        S: Shape + 'static,
        D: Device + 'static,
    {
        buf_any.downcast_ref::<Buffer<T, D, S>>()
    }

    pub fn get_buf_mut<'cache, 'dev, T, D, S>(
        &'cache mut self,
        device: &'dev D,
        id: Ident,
    ) -> Option<&'cache mut Buffer<'dev, T, D, S>>
    where
        T: 'static  + Number,
        S: Shape + 'static,
        D: for<'b> Alloc<'b, T, S> + 'static,
    {
        todo!()
    }

    pub fn get_buf<'cache, 'dev, T, D, S>(
        &'cache mut self,
        device: &'dev D,
        id: Ident,
    ) -> Option<&'cache Buffer<'dev, T, D, S>>
    where
        T: 'static  + Number,
        S: Shape + 'static,
        D: for<'b> Alloc<'b, T, S> + 'static,
    {
        //let cache: Ref<'cache, _> = self.cache.try_borrow().ok()?;

        /*return match self.cache2.get(&id ) {
            None => {self.add_buf(device, id); todo!()},
            Some(buf) => {
                BorrowedCache::downcast_ref_buf::<T, D, S>(buf)
            },
        };*/

        if self.cache2.get(&id).is_none() {
            return self.add_buf(device, id);
        }

        let buf_any = self.cache2.get(&id).unwrap();
        downcast_ref_buf(buf_any)


        /*if let Some(buf) = self.cache2.get(&id) {
            return downcast_ref_buf(buf)
        }*/

        /*match self.cache2.get(&id) {
            Some(buf) => {
                self.downcast_ref_buf::<T, D, S>(buf)
            },
            None => self.add_buf(device, id),
        }*/
    }
}

#[cfg(test)]
mod tests {
    use crate::{Ident, CPU};

    #[test]
    fn test_get_borrowed() {
        let device = CPU::new();

        let mut binding = device
            .cache2
            .borrow_mut();
        let buf = binding
            .get_buf::<f32, _, ()>(&device, Ident::new_bumped(10));
        
        let buf2 = binding
            .get_buf::<f32, _, ()>(&device, Ident::new_bumped(100));

        println!("buf2: {buf2:?}");

    }
}
