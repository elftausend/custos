use crate::{CUDA, CPU, Device, cuda::CUDAPtr, OnDropBuffer, impl_buffer_hook_traits, Buffer, Shape, impl_wrapped_data, WrappedData, OnNewBuffer, HasModules, PtrType, HasId};


pub enum UntypedDevice<Mods> {
    CPU(CPU<Mods>),
    CUDA(CUDA<Mods>),
}

pub enum CpuData {

}

pub enum UntypedData {
    CPU(CpuData)
}

impl PtrType for UntypedData {
    fn size(&self) -> usize {
        todo!()
    }

    fn flag(&self) -> crate::flag::AllocFlag {
        todo!()
    }

    unsafe fn set_flag(&mut self, flag: crate::flag::AllocFlag) {
        todo!()
    }
}

impl HasId for UntypedData {
    fn id(&self) -> crate::Id {
        todo!()
    }
}

pub struct Untyped<Mods> {
    device: UntypedDevice<Mods> 
}

impl<Mods: OnDropBuffer> Device for Untyped<Mods> {
    type Base<T, S: crate::Shape> = UntypedData;
    type Data<T, S: crate::Shape> = Mods::Wrap<T, UntypedData>;
    type Error = ();

    #[inline]
    fn base_to_data<T, S: crate::Shape>(&self, base: Self::Base<T, S>) -> Self::Data<T, S> {
        self.wrap_in_base(base)
    }

    #[inline]
    fn wrap_to_data<T, S: crate::Shape>(&self, wrap: Self::Wrap<T, Self::Base<T, S>>) -> Self::Data<T, S> {
        wrap
    }

    #[inline]
    fn data_as_wrap<'a, T, S: crate::Shape>(
        data: &'a Self::Data<T, S>,
    ) -> &'a Self::Wrap<T, Self::Base<T, S>> {
        data
    }

    #[inline]
    fn data_as_wrap_mut<'a, T, S: crate::Shape>(
        data: &'a mut Self::Data<T, S>,
    ) -> &'a mut Self::Wrap<T, Self::Base<T, S>> {
        data
    }
}

impl<Mods> HasModules<Mods> for Untyped<Mods> {
    #[inline]
    fn modules(&self) -> &Mods {
        match &self.device {
            UntypedDevice::CPU(cpu) => cpu.modules(),
            UntypedDevice::CUDA(cuda) => cuda.modules()
        }
    }
}

impl_wrapped_data!(Untyped);
impl_buffer_hook_traits!(Untyped);