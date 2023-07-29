use core::convert::Infallible;

use super::Device;
use crate::{
    cpu::CPUPtr,
    flag::AllocFlag,
    module_comb::{
        Alloc, Buffer, HasId, HasModules, Module, OnNewBuffer, Retrieve, Retriever, Setup, OnDropBuffer,
    },
    Shape,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CPU<Mods> {
    pub modules: Mods,
}

impl<Mods: OnDropBuffer> Device for CPU<Mods> {
    type Error = Infallible;

    fn new() -> Result<Self, Self::Error> {
        todo!()
        // Ok(CPU::new())
    }
}

impl<Mods: OnDropBuffer> OnDropBuffer for CPU<Mods> {
    #[inline]
    fn on_drop<'a, T, D: Device, S: Shape>(&self, device: &'a D, buf: &Buffer<T, D, S>) {
        self.modules.on_drop(device, buf)
    }
}

impl<Mods> HasModules<Mods> for CPU<Mods> {
    #[inline]
    fn modules(&self) -> &Mods {
        &self.modules
    }
}

impl<SimpleMods> CPU<SimpleMods> {
    #[inline]
    pub fn new<NewMods>() -> CPU<NewMods>
    where
        SimpleMods: Module<CPU<SimpleMods>, Module = NewMods>,
        NewMods: Setup<CPU<NewMods>>,
    {
        let mut cpu = CPU {
            modules: SimpleMods::new(),
        };
        NewMods::setup(&mut cpu);
        cpu
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

impl<T, D: Device, S: Shape, Mods: OnNewBuffer<T, D, S>> OnNewBuffer<T, D, S> for CPU<Mods>
{
    #[inline]
    fn on_new_buffer(&self, device: &D, new_buf: &Buffer<T, D, S>) {
        self.modules.on_new_buffer(device, new_buf)
    }
}

impl<Mods: Retrieve<Self>> Retriever for CPU<Mods> {
    #[inline]
    fn retrieve<T, S: Shape>(&self, len: usize) -> Buffer<T, Self, S> {
        let data = self.modules.retrieve::<T, S>(self, len);
        Buffer {
            data,
            device: Some(self),
            // id: LocationId::new()
        }
    }
}
