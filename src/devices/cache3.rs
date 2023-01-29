use core::{any::Any, mem::transmute};
use std::collections::HashMap;

use crate::{Alloc, Buffer, Device, Ident, Shape, CPU, flag::AllocFlag};

#[derive(Debug, Default)]
pub struct Cache3 {
    buffers: HashMap<Ident, Box<dyn Any>>,
}

impl Cache3 {
    pub fn add_new_buf<'b, T, S, D>(
        &mut self,
        device: &'b D,
        id: Ident,
        callback: fn(),
    ) -> &'b mut Buffer<'b, T, D, S>
    where
        T: 'static,
        S: Shape + 'static,
        D: for<'a> Alloc<'a, T, S> + 'static,
    {
        let buf: Buffer<'static, T, D, S> =
            unsafe { transmute(Buffer::<'b, T, D, S>::new(device, id.len)) };

        self.buffers.insert(id, Box::new(buf));

        callback();

        self.get_mut(device, id, callback)
    }

    #[inline]
    pub fn get_mut<'a, T, D, S>(
        &mut self,
        device: &'a D,
        id: Ident,
        callback: fn(),
    ) -> &'a mut Buffer<'a, T, D, S>
    where
        T: 'static,
        D: for<'b> Alloc<'b, T, S> + 'static,
        S: Shape + 'static,
    {
        match self.buffers.get_mut(&id) {
            Some(cached_buf) => {
                callback();
                
                let cached_buf = cached_buf.downcast_mut::<Buffer<T, D, S>>().expect(
                    "Failed to return Buffer. Type in cache does not match with requested type.",
                );

                unsafe { transmute(cached_buf) }
            }
            None => self.add_new_buf(device, id, callback),
        }
    }

    #[inline]
    pub fn get<'a, T, D, S>(
        &mut self,
        device: &'a D,
        id: Ident,
        callback: fn(),
    ) -> &'a Buffer<'a, T, D, S>
    where
        T: 'static,
        D: for<'b> Alloc<'b, T, S> + 'static,
        S: Shape + 'static,
    {
        self.get_mut(device, id, callback)
    }

    fn get_ref_added<T, D, S>(&mut self, ident: Ident) -> &Buffer<T, D, S>
    where
        T: 'static,
        D: Device + 'static,
        S: Shape + 'static,
    {
        self.buffers
            .get_mut(&ident)
            .unwrap()
            .downcast_mut::<&Buffer<T, D, S>>()
            .expect("Failed to return Buffer. Type in cache does not match with requested type.")
    }
}

impl<'a, T, D: Device, S: Shape> Buffer<'a, T, D, S> {
    pub fn from_slice(device: &'a D, slice: &[T]) {}
}

#[cfg(test)]
mod tests {
    use core::{mem::transmute, marker::PhantomData};

    use crate::{bump_count, set_count, Buffer, Device, Ident, Shape, CPU};

    use super::Cache3;

    pub struct Emit<'a> {
        _p: PhantomData<&'a u8>
    }


    fn test<'a, T, D: Device, S: Shape>(x: &'a Buffer<'a, T, D, S>) -> &'a Buffer<'a, T, D, S> {
        x
    }

    #[test]
    fn test_store_reference() {
        let device = CPU::new();

        let mut cache = Cache3::default();

        let mut buf = Buffer::<f32, CPU, ()>::new(&device, 10);

        let ident = Ident::new(10);
        //    bump_count();

        let static_buf: &Buffer<'static, f32, CPU, ()> = unsafe { transmute(&buf) };
        cache.buffers.insert(ident, Box::new(static_buf));

        let x = cache
            .buffers
            .get(&ident)
            .unwrap()
            .downcast_ref::<&Buffer<f32, CPU>>()
            .unwrap();

        x.len();
    }

    #[test]
    fn test_get() {
        let device = CPU::new();

        let mut cache = Cache3::default();

        let buf = cache.get::<f32, CPU, ()>(&device, Ident::new(10), bump_count);

        set_count(0);

        let buf1 = cache.get::<f32, CPU, ()>(&device, Ident::new(10), bump_count);

        assert_eq!(buf.ptr, buf1.ptr);

        let buf2 = cache.get::<f32, CPU, ()>(&device, Ident::new(10), bump_count);

        assert_ne!(buf.ptr, buf2.ptr);
    }
}
