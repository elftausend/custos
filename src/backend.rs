use core::marker::PhantomData;

use crate::{Base, Setup, Module, LazySetup, OnDropBuffer, cpu::CPUPtr, Device, Alloc, OnNewBuffer, Shape, PtrConv, Retriever, Buffer};

pub struct Backend<Dev, Mods = Base> {
    pub modules: Mods,
    pub device: Dev
}

impl<Dev: Device, Mods: OnDropBuffer> OnDropBuffer for Backend<Dev, Mods> {
    #[inline]
    fn on_drop_buffer<'a, T, D: crate::Device, S: crate::Shape>(&self, device: &'a D, buf: &crate::Buffer<T, D, S>) {
        self.modules.on_drop_buffer(device, buf)
    }
}

impl<Dev: Device, Mods: OnDropBuffer> Device for Backend<Dev, Mods> {
    type Data<T, S: crate::Shape> = Dev::Data<T, S>;
    type Error = Dev::Error;
}

impl<Dev: PtrConv, Mods: OnDropBuffer, OtherMods: OnDropBuffer> PtrConv<Backend<Dev, OtherMods>> for Backend<Dev, Mods> {
    #[inline]
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(
        data: &Self::Data<T, IS>,
        flag: crate::flag::AllocFlag,
    ) -> <Self as Device>::Data<Conv, OS> 
    {
        Dev::convert(data, flag)
    }
}

impl<T, Dev: Device + Alloc<T>, Mods: crate::Retrieve<Self, T>> Retriever<T> for Backend<Dev, Mods> {
    fn retrieve<S, const NUM_PARENTS: usize>(
        &self,
        len: usize,
        parents: impl crate::Parents<NUM_PARENTS>,
    ) -> crate::Buffer<T, Self, S>
    where
        S: Shape 
    {
        let data = self
            .modules
            .retrieve::<S, NUM_PARENTS>(self, len, parents);
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
        T: Clone {
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

impl<T, D: Device, S: Shape, Dev, Mods: OnNewBuffer<T, D, S>> OnNewBuffer<T, D, S> for Backend<Dev, Mods> {
    #[inline]
    fn on_new_buffer(&self, device: &D, new_buf: &crate::Buffer<T, D, S>) {
        self.modules.on_new_buffer(device, new_buf)
    }
}


#[derive(Debug, Default)]
pub struct CPU<SimpleMods = ()> {
    _p: PhantomData<SimpleMods>
}

impl<SimpleMods> CPU<SimpleMods> {
    pub fn new<NewMods>() -> Backend<CPU, NewMods> 
    where
        SimpleMods: Module<CPU, Module = NewMods>,
        NewMods: Setup<CPU>,
    {
        let mut cpu = Backend {
            modules: SimpleMods::new(),
            device: CPU::default() 
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
        T: Clone {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::{Base, Cached, Lazy, Buffer};
    use super::{Backend, CPU};

    #[test]
    fn test_build_device() {
        let device = CPU::<Base>::new();
        Buffer::<f32, _>::new(&device, 10);
    }
}
