mod features;
pub use features::*;

mod modules;
pub use modules::*;

use crate::{cpu::CPUPtr, flag::AllocFlag, Shape, StackArray};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
struct CPU<Mods> {
    modules: Mods,
}

pub trait Device {}

impl<Mods> Device for CPU<Mods> {}

pub trait Alloc {
    type Data<T, S: Shape>;

    fn alloc<T, S: Shape>(&self, len: usize, flag: AllocFlag) -> Self::Data<T, S>;
    fn alloc_from_slice<T, S: Shape>(&self, data: &[T]) -> Self::Data<T, S>
    where
        T: Clone;

    /// If the vector `vec` was allocated previously, this function can be used in order to reduce the amount of allocations, which may be faster than using a slice of `vec`.
    #[inline]
    #[cfg(not(feature = "no-std"))]
    fn alloc_from_vec<T, S: Shape>(&self, vec: Vec<T>) -> Self::Data<T, S>
    where
        T: Clone,
    {
        self.alloc_from_slice(&vec)
    }

    /// Allocates a pointer with the array provided by the `S:`[`Shape`] generic.
    /// By default, the array is flattened and then passed to [`Alloc::with_slice`].
    #[inline]
    fn alloc_from_array<T, S: Shape>(&self, array: S::ARR<T>) -> Self::Data<T, S>
    where
        T: Clone,
    {
        let stack_array = StackArray::<S, T>::from_array(array);
        self.alloc_from_slice(stack_array.flatten())
    }
}

impl<Mods> Alloc for CPU<Mods> {
    type Data<T, S: Shape> = CPUPtr<T>;

    fn alloc<T, S: Shape>(&self, mut len: usize, flag: AllocFlag) -> Self::Data<T, S> {
        assert!(len > 0, "invalid buffer len: 0");

        if S::LEN > len {
            len = S::LEN
        }

        CPUPtr::new_initialized(len, flag)
    }

    fn alloc_from_slice<T, S>(&self, data: &[T]) -> Self::Data<T, S>
    where
        S: Shape,
        T: Clone,
    {
        assert!(!data.is_empty(), "invalid buffer len: 0");
        assert!(S::LEN <= data.len(), "invalid buffer len: {}", data.len());

        let cpu_ptr = unsafe { CPUPtr::new(data.len(), AllocFlag::None) };
        let slice = unsafe { std::slice::from_raw_parts_mut(cpu_ptr.ptr, data.len()) };
        slice.clone_from_slice(data);

        cpu_ptr
    }

    fn alloc_from_vec<T, S: Shape>(&self, mut vec: Vec<T>) -> Self::Data<T, S>
    where
        T: Clone,
    {
        assert!(!vec.is_empty(), "invalid buffer len: 0");

        let ptr = vec.as_mut_ptr();
        let len = vec.len();
        core::mem::forget(vec);

        unsafe { CPUPtr::from_ptr(ptr, len, AllocFlag::None) }
    }
}

pub trait Module<D>  {
    type Module;

    fn new() -> Self::Module;
}


impl<Mods> CPU<Mods> {
    #[inline]
    pub fn new<NewMods>() -> CPU<NewMods> 
    where Mods: Module<CPU<Mods>, Module = NewMods>
    {
        CPU {
            modules: Mods::new()
        }
    }

    pub fn new2<NewMods>() -> CPU<<Mods as Module<CPU<NewMods>>>::Module>
    where
        Mods: Module<CPU<NewMods>>
    {
        /*let res: <Mods as Module<CPU<Mods>>>::Module = <Mods as Module<Self>>::new();
        res*/
        let res: <Mods as Module<CPU<NewMods>>>::Module = <Mods as Module<CPU<NewMods>>>::new();
        
        CPU {
            modules: res
        }
        // let device: CachedModule<Autograd<Base>, CPU<Base>> = <Cached<Autograd::<Base>> as Module::<CPU<Base>>>::new();
    }
}

// use other Retrieve (without _device?) for Devices? (simpler api)
impl<Mods: Retrieve<Self>> Retrieve<Self> for CPU<Mods> {
    #[inline]
    fn retrieve<T, S: Shape>(&self, _device: &Self, len: usize) -> <Self as Alloc>::Data<T, S>
    where
        Self: Alloc 
    {
        self.modules.retrieve(self, len)
    }
}

#[cfg(test)]
mod tests {

    use super::{CPU, Cached, Autograd, Base, Module, CachedModule, Alloc, Retrieve};

    fn take_generic_dev<D: Alloc + Retrieve<D>>(device: &D) {
        device.retrieve::<f32, ()>(device, 10);
    }

    fn take_generic_dev_alloc<D: Alloc>(device: &D) {
        device.alloc::<f32, ()>(10, crate::flag::AllocFlag::None);
    }

    #[test]
    fn test_cpu_creation() {
        let device: CPU<Autograd<Cached<Base>>> = Default::default();
        take_generic_dev(&device);

        // CPU::<Cached<Autograd<Base>>>::new() -> select default type based on build time feature selection?
        let res: CPU<CachedModule<Autograd<Base>, CPU<Cached<Autograd<Base>>>>> = CPU::<Cached<Autograd<Base>>>::new();

        take_generic_dev_alloc(&res);
        take_generic_dev(&res);

        let res: CPU<CachedModule<Autograd<Base>, CPU<Cached<Autograd<Base>>>>> = CPU::<Cached<Autograd<Base>>>::new2();
        // let device: CachedModule<Autograd<Base>, CPU<Base>> = <Cached<Autograd::<Base>> as Module::<CPU<Base>>>::new();
        // let device: CachedModule<Autograd<Base>, CPU<Base>> = Module::<Cached<Autograd<Base>>, CPU<Base>>::new();

        
    }
}
