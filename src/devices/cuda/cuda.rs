use core::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use crate::{
    cuda::{api::cumalloc, CUDAPtr},
    flag::AllocFlag,
    impl_buffer_hook_traits, impl_retriever, pass_down_grad_fn, pass_down_optimize_mem_graph,
    pass_down_tape_actions, Alloc, Base, Buffer, CloneBuf, Device, Module as CombModule,
    OnDropBuffer, OnNewBuffer, PtrConv, Setup, Shape,
};

use super::{
    api::{cuMemcpy, cu_write_async},
    CudaDevice,
};

pub trait IsCuda: Device {}

/// Used to perform calculations with a CUDA capable device.
pub struct CUDA<Mods = Base> {
    pub modules: Mods,
    pub device: CudaDevice,
}

impl_retriever!(CUDA);
impl_buffer_hook_traits!(CUDA);

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
    pub fn new<NewMods>(idx: usize) -> crate::Result<CUDA<NewMods>>
    where
        SimpleMods: CombModule<CUDA, Module = NewMods>,
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

impl<Mods: OnDropBuffer> Device for CUDA<Mods> {
    type Data<T, S: Shape> = CUDAPtr<T>;
    type Base<T, S> = CUDAPtr<T>;
    type Error = i32;

    #[inline(always)]
    fn base_to_data<T, S: Shape>(&self, base: Self::Base<T, S>) -> Self::Data<T, S> {
        self.wrap_in_base(base)
    }

    #[inline(always)]
    fn wrap_to_data<T, S: Shape>(&self, wrap: Self::Wrap<T, Self::Base<T, S>>) -> Self::Data<T, S> {
        wrap
    }
}

impl<Mods: OnDropBuffer, T> Alloc<T> for CUDA<Mods> {
    #[inline]
    fn alloc<S: Shape>(&self, len: usize, flag: crate::flag::AllocFlag) -> Self::Data<T, S> {
        CUDAPtr::new(len, flag)
    }

    fn alloc_from_slice<S: Shape>(&self, data: &[T]) -> Self::Data<T, S>
    where
        T: Clone,
    {
        let ptr = cumalloc::<T>(data.len()).unwrap();
        cu_write_async(ptr, data, &self.mem_transfer_stream).unwrap();
        self.mem_transfer_stream.sync().unwrap();
        CUDAPtr {
            ptr,
            len: data.len(),
            flag: AllocFlag::None,
            p: PhantomData,
        }
    }
}

impl<Mods: OnDropBuffer> IsCuda for CUDA<Mods> {}

#[cfg(feature = "fork")]
impl<Mods> crate::ForkSetup for CUDA<Mods> {
    #[inline]
    fn fork_setup(&mut self) {
        // TODO: maybe check if device supports unified memory
    }
}

pass_down_tape_actions!(CUDA);
pass_down_grad_fn!(CUDA);

impl<Mods: OnDropBuffer, OtherMods: OnDropBuffer> PtrConv<CUDA<OtherMods>> for CUDA<Mods> {
    #[inline]
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(
        ptr: &Self::Data<T, IS>,
        flag: AllocFlag,
    ) -> Self::Data<Conv, OS> {
        CUDAPtr {
            ptr: ptr.ptr,
            len: ptr.len,
            flag,
            p: PhantomData,
        }
    }
}

impl<'a, Mods: OnDropBuffer + OnNewBuffer<T, Self, ()>, T> CloneBuf<'a, T> for CUDA<Mods> {
    fn clone_buf(&'a self, buf: &Buffer<'a, T, CUDA<Mods>>) -> Buffer<'a, T, CUDA<Mods>> {
        let cloned = Buffer::new(self, buf.len());
        unsafe {
            cuMemcpy(
                cloned.ptrs().2,
                buf.ptrs().2,
                buf.len() * std::mem::size_of::<T>(),
            );
        }
        cloned
    }
}

pass_down_optimize_mem_graph!(CUDA);

#[cfg(test)]
mod tests {
    use crate::{Base, Buffer, ClearBuf, Device, Retriever, Shape};

    use super::{IsCuda, CUDA};

    // compile-time isCuda test
    fn take_cu_buffer<T, D: IsCuda + Retriever<T>, S: Shape>(device: &D, buf: &Buffer<T, D, S>) {
        let _buf = device.retrieve::<S, 0>(buf.len(), ());
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
