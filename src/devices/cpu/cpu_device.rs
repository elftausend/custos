use core::{convert::Infallible, ops::DerefMut};

use crate::{
    cpu::CPUPtr, flag::AllocFlag, impl_device_traits, Alloc, Base, Buffer,
    CloneBuf, Device, DevicelessAble, HasModules, IsShapeIndep, Module, OnDropBuffer, OnNewBuffer,
    Setup, Shape, WrappedData, AddLayer, RemoveLayer,
};

pub trait IsCPU {}

/// A CPU is used to perform calculations on the host CPU.
/// To make new operations invocable, a trait providing new functions should be implemented for [CPU].
///
/// # Example
/// ```
/// use custos::{CPU, Read, Buffer, Base, Device};
///
/// let device = CPU::<Base>::new();
/// let a = device.buffer([1, 2, 3]);
/// //let a = Buffer::from((&device, [1, 2, 3]));
///
/// let out = device.read(&a);
///
/// assert_eq!(out, vec![1, 2, 3]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CPU<Mods = Base> {
    pub modules: Mods,
}

impl_device_traits!(CPU);

impl<Mods> IsCPU for CPU<Mods> {}

impl<Mods: OnDropBuffer> Device for CPU<Mods> {
    type Error = Infallible;
    type Base<T, S: Shape> = CPUPtr<T>;
    type Data<T, S: Shape> = Self::Wrap<T, Self::Base<T, S>>;
    // type WrappedData<T, S: Shape> = ;

    fn new() -> Result<Self, Self::Error> {
        todo!()
        // Ok(CPU::<Base>::new())
    }

    #[inline(always)]
    fn base_to_data<T, S: Shape>(&self, base: Self::Base<T, S>) -> Self::Data<T, S> {
        self.wrap_in_base(base)
    }

    #[inline(always)]
    fn wrap_to_data<T, S: Shape>(&self, wrap: Self::Wrap<T, Self::Base<T, S>>) -> Self::Data<T, S> {
        wrap
    }

    #[inline(always)]
    fn data_as_wrap<'a, T, S: Shape>(
        data: &'a Self::Data<T, S>,
    ) -> &'a Self::Wrap<T, Self::Base<T, S>> {
        data
    }

    #[inline(always)]
    fn data_as_wrap_mut<'a, T, S: Shape>(
        data: &'a mut Self::Data<T, S>,
    ) -> &'a mut Self::Wrap<T, Self::Base<T, S>> {
        data
    }

    // #[inline]
    // fn wrap(&self) {}
}

impl<T, S: Shape> DevicelessAble<'_, T, S> for CPU<Base> {}

impl<Mods> HasModules<Mods> for CPU<Mods> {
    #[inline]
    fn modules(&self) -> &Mods {
        &self.modules
    }
}

impl<SimpleMods> CPU<SimpleMods> {
    #[inline]
    pub fn new<NewMods>() -> CPU<SimpleMods::Module>
    where
        SimpleMods: Module<CPU, Module = NewMods>,
        NewMods: Setup<CPU<NewMods>>,
    {
        let mut cpu = CPU {
            modules: SimpleMods::new(),
        };
        NewMods::setup(&mut cpu).unwrap();
        cpu
    }
}

impl<Mods> CPU<Mods> {
    #[inline]
    pub fn add_layer<Mod>(self) -> CPU<Mod::Wrapped> 
    where
        Mod: AddLayer<Mods, CPU>,
    {
        CPU {
            modules: Mod::wrap_layer(self.modules)
        }
    }

    pub fn remove_layer<NewMods>(self) -> CPU::<NewMods>
    where
        Mods: RemoveLayer<NewMods> 
    {
        CPU {
            modules: self.modules.inner_mods()
        }
    }
}

impl<T, Mods: OnDropBuffer> Alloc<T> for CPU<Mods> {
    fn alloc<S: Shape>(&self, mut len: usize, flag: AllocFlag) -> Self::Base<T, S> {
        assert!(len > 0, "invalid buffer len: 0");

        if S::LEN > len {
            len = S::LEN
        }

        // self.wrap_in_base(CPUPtr::new_initialized(len, flag))
        CPUPtr::new_initialized(len, flag)
    }

    fn alloc_from_slice<S>(&self, data: &[T]) -> Self::Base<T, S>
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

    fn alloc_from_vec<S: Shape>(&self, mut vec: Vec<T>) -> Self::Base<T, S>
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

#[cfg(feature = "lazy")]
impl<Mods> crate::LazyRun for CPU<Mods> {}

impl<Mods: crate::RunModule<Self>> crate::Run for CPU<Mods> {
    #[inline]
    unsafe fn run(&self) -> crate::Result<()> {
        self.modules.run(self)
    }
}

#[cfg(feature = "lazy")]
impl<Mods> crate::LazySetup for CPU<Mods> {}

#[cfg(feature = "fork")]
impl<Mods> crate::ForkSetup for CPU<Mods> {}

impl<'a, Mods: OnDropBuffer + OnNewBuffer<T, Self, S>, T: Clone, S: Shape> CloneBuf<'a, T, S>
    for CPU<Mods>
where
    Self::Data<T, S>: DerefMut<Target = [T]>,
{
    #[inline]
    fn clone_buf(&'a self, buf: &Buffer<'a, T, CPU<Mods>, S>) -> Buffer<'a, T, CPU<Mods>, S> {
        let mut cloned = Buffer::new(self, buf.len());
        cloned.clone_from_slice(buf);
        cloned
    }
}

unsafe impl<Mods: OnDropBuffer> IsShapeIndep for CPU<Mods> {}

#[cfg(test)]
mod tests {
    use crate::{CPU, Base};

    #[test]
    fn test_add_layer() {
        let cpu = CPU::<Base>::new();
        let cpu = cpu.add_layer::<crate::Cached<()>>();
        let cpu = cpu.add_layer::<crate::Fork<()>>();

        let _cpu = cpu.remove_layer();
    }
}
