use core::{cell::RefMut, panic::Location};
use std::collections::HashMap;

use std::rc::Rc;

use crate::{bump_count, flag::AllocFlag, Alloc, Buffer, Device, Ident, PtrConv, Shape, CPU};

pub trait CallerCacheReturn {
    /// Returns a reference to a device's [`Cache`].
    fn cache(&self) -> core::cell::Ref<TrackCallerCache<Self>>
    where
        Self: PtrConv;

    /// Returns a mutable reference to a device's [`Cache`].
    fn cache_mut(&self) -> RefMut<TrackCallerCache<Self>>
    where
        Self: PtrConv;
}
pub trait Cache: Device {
    type CallerCache: TrackCallerCacheAble<Self>;

    #[track_caller]
    #[inline]
    fn call<T, S: Shape>(&self, len: usize) -> Buffer<T, Self, S>
    where
        for<'b> Self: Alloc<'b, T, S>,
    {
        Self::CallerCache::get(self, len)
    }
}

impl Cache for CPU {
    type CallerCache = TrackCallerCache<CPU>;
}

#[derive(Debug, Default)]
pub struct TrackCallerCache<D: Device> {
    nodes: HashMap<&'static std::panic::Location<'static>, Rc<D::Ptr<u8, ()>>>,
}

pub trait TrackCallerCacheAble<D: Device> {
    #[track_caller]
    fn get<T, S: Shape>(device: &D, len: usize) -> Buffer<T, D, S>
    where
        for<'b> D: Alloc<'b, T, S>;
}

impl<D: Device> TrackCallerCacheAble<D> for () {
    #[inline]
    fn get<T, S: Shape>(device: &D, len: usize) -> Buffer<T, D, S>
    where
        for<'b> D: Alloc<'b, T, S>,
    {
        Buffer::new(device, len)
    }
}

impl<D: Device> TrackCallerCacheAble<D> for TrackCallerCache<D>
where
    D: PtrConv + CallerCacheReturn,
{
    #[track_caller]
    fn get<T, S: Shape>(device: &D, len: usize) -> Buffer<T, D, S>
    where
        D: for<'a> Alloc<'a, T, S>,
    {
        device.cache_mut().get(device, Ident::new(len), bump_count)
    }
}

impl<D: PtrConv> TrackCallerCache<D> {
    #[track_caller]
    pub fn get<'a, T, S>(
        &mut self,
        device: &'a D,
        ident: Ident,
        callback: fn(),
    ) -> Buffer<'a, T, D, S>
    where
        D: Alloc<'a, T, S>,
        S: Shape,
    {
        let maybe_allocated = self.nodes.get(Location::caller());

        match maybe_allocated {
            Some(ptr) => {
                callback();
                let typed_ptr = unsafe { D::convert(ptr, AllocFlag::Wrapper) };

                Buffer {
                    ptr: typed_ptr,
                    device: Some(device),
                    ident: Some(ident),
                }
            }
            None => self.add_node(device, ident, callback),
        }
    }

    #[track_caller]
    fn add_node<'a, T, S>(
        &mut self,
        device: &'a D,
        ident: Ident,
        callback: fn(),
    ) -> Buffer<'a, T, D, S>
    where
        D: Alloc<'a, T, S>,
        S: Shape,
    {
        let ptr = device.alloc(ident.len, AllocFlag::Wrapper);

        let untyped_ptr = unsafe { D::convert(&ptr, AllocFlag::None) };
        self.nodes.insert(Location::caller(), Rc::new(untyped_ptr));

        callback();

        Buffer {
            ptr,
            device: Some(device),
            ident: Some(Ident {
                idx: ident.idx,
                len: ident.len,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use core::ops::Add;

    use crate::{devices::caller_cache::CallerCacheReturn, Buffer, Device, CPU};

    use super::Cache;

    #[track_caller]
    fn add<'a, T: Add<Output = T> + Copy>(
        device: &'a CPU,
        lhs: &Buffer<T>,
        rhs: &Buffer<T>,
    ) -> Buffer<'a, T> {
        let len = std::cmp::min(lhs.len(), rhs.len());

        let mut out = device.call::<T, ()>(len);

        for idx in 0..len {
            out[idx] = lhs[idx] + rhs[idx];
        }

        out
    }
    #[test]
    fn test_caller_cache() {
        let device = CPU::new();

        let lhs = device.buffer([1, 2, 3, 4]);
        let rhs = device.buffer([1, 2, 3, 4]);

        for _i in 0..100 {
            add(&device, &lhs, &rhs);
        }

        assert_eq!(device.cache().nodes.len(), 1);
    }
}
