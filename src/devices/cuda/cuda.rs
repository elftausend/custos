use core::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use crate::{
    Alloc, Base, Buffer, CloneBuf, Device, IsShapeIndep, Module as CombModule, OnNewBuffer, Setup,
    Shape, Unit, WrappedData,
    cuda::{CUDAPtr, api::cumalloc},
    flag::AllocFlag,
    impl_device_traits,
};

use super::{
    CudaDevice,
    api::{cu_write_async, cuMemcpy},
};

pub trait IsCuda: Device {}

/// Used to perform calculations with a CUDA capable device.
pub struct CUDA<Mods = Base> {
    pub modules: Mods,
    pub device: CudaDevice,
}

impl_device_traits!(CUDA);

impl<Mods> Deref for CUDA<Mods> {
    type Target = CudaDevice;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl<Mods> DerefMut for CUDA<Mods> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.device
    }
}

// TODO: convert to device with other mods

impl<SimpleMods> CUDA<SimpleMods> {
    /// Returns an [CUDA] device at the specified device index.
    /// # Errors
    /// - No device was found at the given device index
    /// - some other CUDA related errors
    #[inline]
    pub fn new<'a, NewMods>(idx: usize) -> crate::Result<CUDA<NewMods>>
    where
        Self: 'a,
        SimpleMods: CombModule<'a, CUDA, Module = NewMods>,
        NewMods: Setup<CUDA<NewMods>>,
    {
        let mut cuda = CUDA {
            modules: SimpleMods::new(),
            device: CudaDevice::new(idx)?,
        };

        NewMods::setup(&mut cuda)?;

        Ok(cuda)
    }
}

impl<Mods: WrappedData> Device for CUDA<Mods> {
    type Data<'a, T: Unit, S: Shape> = Mods::Wrap<'a, T, CUDAPtr<T>>;
    type Base<T: Unit, S: Shape> = CUDAPtr<T>;
    type Error = i32;

    #[inline(always)]
    fn default_base_to_data<'a, T: Unit, S: Shape>(
        &'a self,
        base: Self::Base<T, S>,
    ) -> Self::Data<'a, T, S> {
        self.wrap_in_base(base)
    }

    #[inline(always)]
    fn default_base_to_data_unbound<'a, T: Unit, S: Shape>(
        &self,
        base: Self::Base<T, S>,
    ) -> Self::Data<'a, T, S> {
        self.wrap_in_base_unbound(base)
    }

    #[inline(always)]
    fn wrap_to_data<'a, T: Unit, S: Shape>(
        &self,
        wrap: Self::Wrap<'a, T, Self::Base<T, S>>,
    ) -> Self::Data<'a, T, S> {
        wrap
    }

    #[inline(always)]
    fn data_as_wrap<'a, 'b, T: Unit, S: Shape>(
        data: &'b Self::Data<'a, T, S>,
    ) -> &'b Self::Wrap<'a, T, Self::Base<T, S>> {
        data
    }

    #[inline(always)]
    fn data_as_wrap_mut<'a, 'b, T: Unit, S: Shape>(
        data: &'b mut Self::Data<'a, T, S>,
    ) -> &'b mut Self::Wrap<'a, T, Self::Base<T, S>> {
        data
    }
}

impl<Mods: WrappedData, T: Unit> Alloc<T> for CUDA<Mods> {
    #[inline]
    fn alloc<S: Shape>(
        &self,
        len: usize,
        flag: crate::flag::AllocFlag,
    ) -> crate::Result<Self::Base<T, S>> {
        Ok(CUDAPtr::new(len, flag)?)
    }

    fn alloc_from_slice<S: Shape>(&self, data: &[T]) -> crate::Result<Self::Base<T, S>>
    where
        T: Clone,
    {
        let ptr = cumalloc::<T>(data.len())?;
        cu_write_async(ptr, data, &self.mem_transfer_stream)?;
        self.mem_transfer_stream.sync()?;
        Ok(CUDAPtr {
            ptr,
            len: data.len(),
            flag: AllocFlag::None,
            p: PhantomData,
        })
    }
}

unsafe impl<Mods: WrappedData> IsShapeIndep for CUDA<Mods> {}

impl<Mods: WrappedData> IsCuda for CUDA<Mods> {}

#[cfg(feature = "fork")]
impl<Mods> crate::ForkSetup for CUDA<Mods> {
    #[inline]
    fn fork_setup(&mut self) {
        // TODO: maybe check if device supports unified memory
    }
}

impl<'a, Mods: WrappedData + OnNewBuffer<'a, T, Self, ()>, T: Unit> CloneBuf<'a, T> for CUDA<Mods> {
    fn clone_buf(&'a self, buf: &Buffer<'a, T, CUDA<Mods>>) -> Buffer<'a, T, CUDA<Mods>> {
        let cloned = Buffer::new(self, buf.len());
        unsafe {
            cuMemcpy(
                cloned.cu_ptr(),
                buf.cu_ptr(),
                buf.len() * std::mem::size_of::<T>(),
            );
        }
        cloned
    }
}

#[cfg(test)]
mod tests {
    use crate::{Base, Buffer, ClearBuf, Device, Retriever, Shape, Unit};

    use super::{CUDA, IsCuda};

    // compile-time isCuda test
    fn take_cu_buffer<'a, T: Unit, D: IsCuda + Retriever<'a, T>, S: Shape>(
        device: &'a D,
        buf: &Buffer<'a, T, D, S>,
    ) {
        let _buf = device.retrieve::<0>(buf.len(), ());
    }

    #[test]
    fn test_cu_buffer_fn() {
        let device = CUDA::<Base>::new(0).unwrap();
        let buf = Buffer::<f32, _, ()>::new(&device, 10);
        take_cu_buffer(&device, &buf)
    }

    #[test]
    #[ignore = "does not work at the moment"]
    fn test_cross_distinct_devices() {
        let dev1 = CUDA::<Base>::new(0).unwrap();
        let mut buf1 = dev1.buffer([1, 2, 3, 4, 5, 6]);

        let dev2 = CUDA::<Base>::new(0).unwrap();
        let mut buf2 = dev1.buffer([1, 2, 3, 4, 5, 6]);

        dev2.clear(&mut buf1);
        dev1.clear(&mut buf2);

        println!("fin");
        assert_eq!(buf1.read(), [0; 6]);
        assert_eq!(buf2.read(), [0; 6]);
    }
}
