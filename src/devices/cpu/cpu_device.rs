use core::{convert::Infallible, ops::DerefMut};

use crate::{
    cpu::CPUPtr, flag::AllocFlag, impl_buffer_hook_traits, impl_retriever, impl_wrapped_data,
    pass_down_grad_fn, pass_down_optimize_mem_graph, pass_down_replace_buf, pass_down_tape_actions,
    Alloc, Base, Buffer, CloneBuf, Device, DevicelessAble, HasModules, Module, OnDropBuffer,
    OnNewBuffer, PtrConv, Setup, Shape, WrappedData,
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

impl<Mods: OnDropBuffer> Device for CPU<Mods> {
    type Error = Infallible;
    type Base<T, S> = CPUPtr<T>;
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

impl_wrapped_data!(CPU);

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

#[cfg(feature = "cached")]
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

pass_down_replace_buf!(CPU);

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

// impl for all devices
impl<Mods: OnDropBuffer, OtherMods: OnDropBuffer> PtrConv<CPU<OtherMods>> for CPU<Mods> {
    #[inline]
    unsafe fn convert<T, IS: Shape, Conv, OS: Shape>(
        data: &Mods::Wrap<T, CPUPtr<T>>,
        flag: AllocFlag,
    ) -> OtherMods::Wrap<Conv, CPUPtr<Conv>> {
        // data.flag()
        todo!()
        // CPUPtr {
        //     ptr: data.ptr as *mut Conv,
        //     len: data.len,
        //     flag,
        //     align: Some(align_of::<T>()),
        //     size: Some(size_of::<T>()),
        // }
    }
}
