use core::marker::PhantomData;

use crate::{
    cpu::CPUPtr, Alloc, Base, Buffer, Device, LazySetup, Module, OnDropBuffer, OnNewBuffer,
    PtrConv, Read, Retriever, Setup, Shape, TapeActions,
};

pub struct Backend<Dev, Mods = Base> {
    pub modules: Mods,
    pub device: Dev,
}

pub trait HasDevice {
    type Dev: Device;
}

impl<Dev, Mods> core::ops::Deref for Backend<Dev, Mods> {
    type Target = Dev;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl<Dev: Device, Mods> HasDevice for Backend<Dev, Mods> {
    type Dev = Dev;
}

pub trait BindDevice<Dev>: Device {
    fn device(&self) -> &Dev;
}

impl<Dev: Device, Mods: OnDropBuffer> BindDevice<Dev> for Backend<Dev, Mods> {
    #[inline]
    fn device(&self) -> &Dev {
        &self.device
    }
}

impl<Dev: Device, Mods: OnDropBuffer> OnDropBuffer for Backend<Dev, Mods> {
    #[inline]
    fn on_drop_buffer<'a, T, D: crate::Device, S: crate::Shape>(
        &self,
        device: &'a D,
        buf: &crate::Buffer<T, D, S>,
    ) {
        self.modules.on_drop_buffer(device, buf)
    }
}

impl<Dev: Device, Mods: OnDropBuffer> Device for Backend<Dev, Mods> {
    type Data<T, S: crate::Shape> = Dev::Data<T, S>;
    type Error = Dev::Error;
}

impl<Dev: PtrConv, Mods: OnDropBuffer, OtherMods: OnDropBuffer> PtrConv<Backend<Dev, OtherMods>>
    for Backend<Dev, Mods>
{
    #[inline]
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(
        data: &Self::Data<T, IS>,
        flag: crate::flag::AllocFlag,
    ) -> <Self as Device>::Data<Conv, OS> {
        Dev::convert(data, flag)
    }
}

impl<Dev: PtrConv, Mods: OnDropBuffer /*OtherMods: OnDropBuffer*/> PtrConv<Dev>
    for Backend<Dev, Mods>
{
    #[inline]
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(
        data: &Self::Data<T, IS>,
        flag: crate::flag::AllocFlag,
    ) -> <Self as Device>::Data<Conv, OS> {
        Dev::convert(data, flag)
    }
}

impl<T, Dev: Device + Alloc<T>, Mods: crate::Retrieve<Self, T>> Retriever<T>
    for Backend<Dev, Mods>
{
    fn retrieve<S, const NUM_PARENTS: usize>(
        &self,
        len: usize,
        parents: impl crate::Parents<NUM_PARENTS>,
    ) -> crate::Buffer<T, Self, S>
    where
        S: Shape,
    {
        let data = self.modules.retrieve::<S, NUM_PARENTS>(self, len, parents);
        let buf = Buffer {
            data,
            device: Some(self),
        };
        self.modules.on_retrieve_finish(&buf);
        buf
    }
}

impl<T, Dev: Alloc<T>, Mods: OnDropBuffer> Alloc<T> for Backend<Dev, Mods> {
    #[inline]
    fn alloc<S: crate::Shape>(&self, len: usize, flag: crate::flag::AllocFlag) -> Self::Data<T, S> {
        self.device.alloc(len, flag)
    }

    #[inline]
    fn alloc_from_slice<S: crate::Shape>(&self, data: &[T]) -> Self::Data<T, S>
    where
        T: Clone,
    {
        self.device.alloc_from_slice(data)
    }

    #[inline]
    fn alloc_from_vec<S: crate::Shape>(&self, vec: Vec<T>) -> Self::Data<T, S>
    where
        T: Clone,
    {
        self.device.alloc_from_slice(&vec)
    }

    #[inline]
    fn alloc_from_array<S: crate::Shape>(&self, array: S::ARR<T>) -> Self::Data<T, S>
    where
        T: Clone,
    {
        self.device.alloc_from_array(array)
    }
}

impl<T, D: Device, S: Shape, Dev, Mods: OnNewBuffer<T, D, S>> OnNewBuffer<T, D, S>
    for Backend<Dev, Mods>
{
    #[inline]
    fn on_new_buffer(&self, device: &D, new_buf: &crate::Buffer<T, D, S>) {
        self.modules.on_new_buffer(device, new_buf)
    }
}

impl<Dev: Read<T, S, Self>, Mods: OnDropBuffer, T, S: Shape> Read<T, S> for Backend<Dev, Mods> {
    type Read<'a> = Dev::Read<'a>
    where
        T: 'a,
        Self: 'a,
        S: 'a;

    #[inline]
    fn read<'a>(&self, buf: &'a Buffer<T, Self, S>) -> Self::Read<'a> {
        Dev::read(&self.device, buf)
    }

    #[inline]
    fn read_to_vec(&self, buf: &Buffer<T, Self, S>) -> Vec<T>
    where
        T: Default + Clone,
    {
        Dev::read_to_vec(&self.device, buf)
    }
}

impl<Dev, Mods: TapeActions<Dev>> TapeActions<Dev> for Backend<Dev, Mods> {
    #[inline]
    fn tape(&self) -> Option<core::cell::Ref<crate::Tape<Dev>>> {
        self.modules.tape()
    }

    #[inline]
    fn tape_mut(&self) -> Option<core::cell::RefMut<crate::Tape<Dev>>> {
        self.modules.tape_mut()
    }
}

#[derive(Debug, Default)]
pub struct CPU<SimpleMods = ()> {
    _p: PhantomData<SimpleMods>,
}

impl<SimpleMods> CPU<SimpleMods> {
    pub fn new<NewMods>() -> Backend<CPU, NewMods>
    where
        SimpleMods: Module<CPU, Module = NewMods>,
        NewMods: Setup<CPU>,
    {
        let mut cpu = Backend {
            modules: SimpleMods::new(),
            device: CPU::default(),
        };
        NewMods::setup(&mut cpu.device);
        cpu
    }
}

impl OnDropBuffer for CPU {}

impl crate::Device for CPU {
    type Data<T, S: crate::Shape> = CPUPtr<T>;

    type Error = ();
}

impl LazySetup for CPU {
    fn lazy_setup(&mut self) {}
}

impl<T> Alloc<T> for CPU {
    fn alloc<S: Shape>(&self, len: usize, flag: crate::flag::AllocFlag) -> Self::Data<T, S> {
        todo!()
    }

    fn alloc_from_slice<S: Shape>(&self, data: &[T]) -> Self::Data<T, S>
    where
        T: Clone,
    {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::{Backend, CPU};
    use crate::{Base, Buffer, Cached, Lazy};

    #[test]
    fn test_build_device() {
        let device = CPU::<Base>::new();
        Buffer::<f32, _>::new(&device, 10);
    }
}
