use core::{convert::Infallible, marker::PhantomData};

use crate::{
    backend::Backend, cpu::CPUPtr, flag::AllocFlag, impl_buffer_hook_traits, impl_retriever, Alloc,
    Base, Buffer, Cached, CachedModule, CloneBuf, Device, DevicelessAble, HasModules, MainMemory,
    Module, OnDropBuffer, OnNewBuffer, Setup, Shape, TapeActions,
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
pub struct CPU<Mods = ()> {
    pub _mods: PhantomData<Mods>,
}

// impl_retriever!(CPU);
// impl_buffer_hook_traits!(CPU);

impl<Mods> IsCPU for CPU<Mods> {}

// maybe
/*impl<Mods: OnDropBuffer> CPU<Mods> {
    pub fn default() -> CPU<CachedModule<Base, CPU<Cached<Base>>>> {
        CPU::<Cached<Base>>::new()
    }
}*/

impl<Mods> Device for CPU<Mods> {
    type Error = Infallible;
    type Data<T, S: Shape> = CPUPtr<T>;

    fn new() -> Result<Self, Self::Error> {
        todo!()
        // Ok(CPU::<Base>::new())
    }
}

impl<Mods> OnDropBuffer for CPU<Mods> {}
impl<Mods, T, D: Device, S: Shape> OnNewBuffer<T, D, S> for CPU<Mods> {}

impl<T, S: Shape> DevicelessAble<'_, T, S> for CPU {}

impl<Mods: OnDropBuffer> MainMemory for Backend<CPU, Mods> {
    #[inline]
    fn as_ptr<T, S: Shape>(ptr: &Self::Data<T, S>) -> *const T {
        ptr.ptr
    }

    #[inline]
    fn as_ptr_mut<T, S: Shape>(ptr: &mut Self::Data<T, S>) -> *mut T {
        ptr.ptr
    }
}

impl MainMemory for CPU {
    #[inline]
    fn as_ptr<T, S: Shape>(ptr: &Self::Data<T, S>) -> *const T {
        ptr.ptr
    }

    #[inline]
    fn as_ptr_mut<T, S: Shape>(ptr: &mut Self::Data<T, S>) -> *mut T {
        ptr.ptr
    }
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

impl<T> Alloc<T> for CPU {
    fn alloc<S: Shape>(&self, mut len: usize, flag: AllocFlag) -> Self::Data<T, S> {
        assert!(len > 0, "invalid buffer len: 0");

        if S::LEN > len {
            len = S::LEN
        }

        CPUPtr::new_initialized(len, flag)
    }

    fn alloc_from_slice<S>(&self, data: &[T]) -> Self::Data<T, S>
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

    fn alloc_from_vec<S: Shape>(&self, mut vec: Vec<T>) -> Self::Data<T, S>
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

/*impl<Mods: TapeActions<D>, D> TapeActions<D> for CPU<Mods> {
    #[inline]
    fn tape(&self) -> Option<core::cell::Ref<crate::Tape<D>>> {
        self.modules.tape()
    }

    #[inline]
    fn tape_mut(&self) -> Option<core::cell::RefMut<crate::Tape<D>>> {
        self.modules.tape_mut()
    }
}*/

#[cfg(feature = "lazy")]
impl<Mods> crate::LazySetup for CPU<Mods> {}

impl<'a, /*Mods: OnDropBuffer + OnNewBuffer<T, Self, S>,*/ T: Clone, S: Shape> CloneBuf<'a, T, S>
    for CPU
{
    #[inline]
    fn clone_buf(&'a self, buf: &Buffer<'a, T, CPU, S>) -> Buffer<'a, T, CPU, S> {
        let mut cloned = Buffer::new(self, buf.len());
        cloned.clone_from_slice(buf);
        cloned
    }
}
