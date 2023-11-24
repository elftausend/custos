use core::{
    convert::Infallible,
    mem::{align_of, size_of},
};

use crate::{
    cpu::CPUPtr, flag::AllocFlag, impl_buffer_hook_traits, impl_retriever, pass_down_grad_fn,
    pass_down_optimize_mem_graph, pass_down_tape_actions, Alloc, Base, Buffer, CloneBuf, Device,
    DevicelessAble, HasModules, Module, OnDropBuffer, OnNewBuffer, PtrConv, Setup, Shape,
    WrappedData,
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

impl_retriever!(CPU);
impl_buffer_hook_traits!(CPU);

impl<Mods> IsCPU for CPU<Mods> {}

impl<Mods: OnDropBuffer + Module<CPU>> WrappedData for CPU<Mods> {
    // type WrappedData<T, S: Shape> = Mods::Data<T, S>;
}

impl<Mods: OnDropBuffer> Device for CPU<Mods> {
    type Error = Infallible;
    type Data<T, S: Shape> = CPUPtr<T>;
    // type WrappedData<T, S: Shape> = ;

    fn new() -> Result<Self, Self::Error> {
        todo!()
        // Ok(CPU::<Base>::new())
    }
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
    pub fn new<NewMods>() -> CPU<NewMods>
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

impl<T, Mods: OnDropBuffer> Alloc<T> for CPU<Mods> {
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

pass_down_optimize_mem_graph!(CPU);
pass_down_grad_fn!(CPU);

#[cfg(feature = "lazy")]
impl<Mods> crate::LazyRun for CPU<Mods> {}

impl<Mods: crate::RunModule<Self>> crate::Run for CPU<Mods> {
    #[inline]
    unsafe fn run(&self) -> crate::Result<()> {
        self.modules.run(self)
    }
}

pass_down_tape_actions!(CPU);

#[cfg(feature = "lazy")]
impl<Mods> crate::LazySetup for CPU<Mods> {}

#[cfg(feature = "fork")]
impl<Mods> crate::ForkSetup for CPU<Mods> {}

impl<'a, Mods: OnDropBuffer + OnNewBuffer<T, Self>, T: Clone, S: Shape> CloneBuf<'a, T, S>
    for CPU<Mods>
{
    #[inline]
    fn clone_buf(&'a self, buf: &Buffer<'a, T, CPU<Mods>, S>) -> Buffer<'a, T, CPU<Mods>, S> {
        let mut cloned = Buffer::new(self, buf.len());
        cloned.clone_from_slice(buf);
        cloned
    }
}

// impl for all devices
impl<Mods: OnDropBuffer, OtherMods: OnDropBuffer> PtrConv<CPU<OtherMods>> for CPU<Mods> {
    #[inline]
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(
        data: &CPUPtr<T>,
        flag: AllocFlag,
    ) -> CPUPtr<Conv> {
        CPUPtr {
            ptr: data.ptr as *mut Conv,
            len: data.len,
            flag,
            align: Some(align_of::<T>()),
            size: Some(size_of::<T>()),
        }
    }
}
