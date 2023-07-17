use core::{cell::RefCell, marker::PhantomData, panic::Location, ops::BitXor, hash::BuildHasherDefault};
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
                nodes_faster: Default::default(),
            }),
        }
    }
}

pub struct CachedModule<Mods, D: Alloc> {
    modules: Mods,
    cache: RefCell<Cache<D>>,
}

#[derive(Default)]
pub struct LocationHasher {
    hash: u64,
}

const K: u64 = 0x517cc1b727220a95;

impl std::hash::Hasher for LocationHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash as u64
    }

    #[inline]
    fn write(&mut self, _bytes: &[u8]) {
        unimplemented!("LocationHasher only hashes u64, (u32 and usize as u64 cast).")
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.hash = self.hash.rotate_left(5).bitxor(i).wrapping_mul(K);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.write_u64(i as u64);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
}

#[derive(Debug, Clone, Copy, Eq)]
pub struct HashLocation<'a> {
    file: &'a str,
    line: u32,
    col: u32,
}

impl PartialEq for HashLocation<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // filename pointer is actually actually unique, then this works (added units tests to check this... still not sure)
        if self.file.as_ptr() != other.file.as_ptr() {
            return false;
        }
        self.line == self.line &&
        self.col == self.col
    }
}

impl<'a> std::hash::Hash for HashLocation<'a> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.file.as_ptr().hash(state);
        let line_col = (self.line as u64) << 9 | self.col as u64;
        line_col.hash(state);
    }
}

impl<'a> From<&'a Location<'a>> for HashLocation<'a> {
    #[inline]
    fn from(loc: &'a Location<'a>) -> Self {
        Self {
            file: loc.file(),
            line: loc.line(),
            col: loc.column(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cache<D: Alloc> {
    // improve hashing speed for location?
    pub nodes: HashMap<&'static Location<'static>, Rc<D::Data<u8, ()>>>,
    pub nodes_faster: HashMap<HashLocation<'static>, Rc<D::Data<u8, ()>>, BuildHasherDefault<LocationHasher>>
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
        let maybe_allocated = self.nodes_faster.get(&Location::caller().into());
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
        self.nodes_faster.insert(Location::caller().into(), Rc::new(untyped_ptr));

        callback();

        data
    }
}

impl<Mods, D: Alloc + PtrConv<SimpleDevice>, SimpleDevice: Alloc + PtrConv<D>> Retrieve<D>
    for CachedModule<Mods, SimpleDevice>
{
    #[inline]
    fn retrieve<T, S: crate::Shape>(&self, device: &D, len: usize) -> D::Data<T, S> {
        self.cache.borrow_mut().get(device, len, || ())
    }
}

#[macro_export]
macro_rules! debug_assert_tracked {
    () => {
        #[cfg(debug_assertions)]
        {
            let location = core::panic::Location::caller();
            assert_ne!((file!(), line!(), column!()), (location.file(), location.line(), location.column()), "Function and operation must be annotated with `#[track_caller]`.");
        }
    };
}

/// This macro is nothing but a mechanism to ensure that the specific operation is annotated with `#[track_caller]`.
/// If the operation is not annotated with `#[track_caller]`, then the macro will cause a panic (in debug mode).
/// 
/// This macro turns the device, length and optionally type information into the following line of code:
/// ## From:
/// ```ignore
/// retrieve!(device, 10, f32)
/// ```
/// ## To:
/// ```ignore
/// custos::debug_assert_tracked!();
/// device.retrieve::<f32, ()>(10)
/// ```
/// 
/// If you ensure that the operation is annotated with `#[track_caller]`, then you can just write the following:
/// ```ignore
/// device.retrieve::<f32, ()>(10)
/// ```
/// 
/// # Example
/// Operation is not annotated with `#[track_caller]` and therefore will panic:
/// ```should_panic
/// use custos::{retrieve, module_comb::{CPU, Retriever, Buffer, Retrieve, Cached, Base}};
/// 
/// fn add_bufs<Mods: Retrieve<CPU<Mods>>>(device: &CPU<Mods>) -> Buffer<f32, CPU<Mods>, ()> {
///     retrieve!(device, 10, f32)
/// }
/// 
/// let device = CPU::<Cached<Base>>::new();
/// add_bufs(&device);
/// ```
/// Operation is annotated with `#[track_caller]`:
/// ```
/// use custos::{Dim1, retrieve, module_comb::{CPU, Retriever, Buffer, Retrieve, Cached, Base}};
/// 
/// #[track_caller]
/// fn add_bufs<Mods: Retrieve<CPU<Mods>>>(device: &CPU<Mods>) -> Buffer<f32, CPU<Mods>, Dim1<30>> {
///     retrieve!(device, 10, f32, Dim1<30>); // you can also specify the shape
///     retrieve!(device, 10) // or infer the type and shape from the output type
/// }
/// 
/// let device = CPU::<Cached<Base>>::new();
/// add_bufs(&device);
/// ```
#[macro_export]
macro_rules! retrieve {
    ($device:ident, $len:expr) => {{
        $crate::debug_assert_tracked!();
        $device.retrieve($len)
    }};
    ($device:ident, $len:expr, $dtype:ty) => {{
        $crate::debug_assert_tracked!();
        $device.retrieve::<$dtype, ()>($len)
    }};
    ($device:ident, $len:expr, $dtype:ty, $shape:ty) => {{
        $crate::debug_assert_tracked!();
        $device.retrieve::<$dtype, $shape>($len)
    }};
}

#[cfg(test)]
mod tests {
    use core::{panic::Location, ptr::addr_of};

    // crate::modules
    use crate::module_comb::{CPU, Base, Retriever, Buffer, Retrieve, location};

    use super::Cached;

    // forgot to add track_caller
    #[cfg(debug_assertions)]
    fn add_bufs<Mods: Retrieve<CPU<Mods>>>(device: &CPU<Mods>) -> Buffer<f32, CPU<Mods>, ()> {
        retrieve!(device, 10, f32)
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn test_forgot_track_caller_runtime_detection() {
        let device = CPU::<Cached<Base>>::new();
        
        let _out = add_bufs(&device);
        let _out = add_bufs(&device);
    }

    #[track_caller]
    fn add_bufs_tracked<Mods: Retrieve<CPU<Mods>>>(device: &CPU<Mods>) -> Buffer<f32, CPU<Mods>, ()> {
        retrieve!(device, 10, f32)
    }

    #[test]
    fn test_added_track_caller() {
        let device = CPU::<Cached<Base>>::new();
        
        let _out = add_bufs_tracked(&device);
        let _out = add_bufs_tracked(&device);
    }

    #[test]
    fn test_location_ref_unique() {
        let ptr = location();
        let ptr1 = location();
        // bad
        assert_ne!(addr_of!(ptr), addr_of!(ptr1));
    }

    #[track_caller]
    fn location_tracked() -> &'static Location<'static> {
        Location::caller()
    }

    #[test]
    fn test_location_file_ptr_unique() {
        let ptr = location();
        let ptr1 = location();
        // good
        assert_eq!(ptr.file().as_ptr(), ptr1.file().as_ptr());
    }

    #[test]
    fn test_location_file_tracked_ptr_unique() {
        let ptr = location_tracked();
        let ptr1 = location_tracked();
        // good
        assert_eq!(ptr.file().as_ptr(), ptr1.file().as_ptr());
    }

    #[test]
    fn test_location_with_different_file_location_ptr_unique() {
        let ptr = location_tracked();
        let ptr1 = location();
        // good
        assert_ne!(ptr.file().as_ptr(), ptr1.file().as_ptr());
    }

    #[test]
    fn test_shift() {
        let (line, col) = (line!(), column!());
        let res = line << 8 | col;
        println!("res: {res}");
    }

}
