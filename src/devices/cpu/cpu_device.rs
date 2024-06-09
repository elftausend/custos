use core::convert::Infallible;

use crate::{
    cpu::CPUPtr, flag::AllocFlag, impl_device_traits, AddLayer, Alloc, Base, Buffer, CloneBuf,
    Device, DevicelessAble, HasModules, IsShapeIndep, Module, OnDropBuffer, OnNewBuffer,
    RemoveLayer, Resolve, Setup, Shape, UnaryFusing, WrappedData,
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
/// let out = a.read();
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
    fn data_as_wrap<T, S: Shape>(data: &Self::Data<T, S>) -> &Self::Wrap<T, Self::Base<T, S>> {
        data
    }

    #[inline(always)]
    fn data_as_wrap_mut<T, S: Shape>(
        data: &mut Self::Data<T, S>,
    ) -> &mut Self::Wrap<T, Self::Base<T, S>> {
        data
    }

    // #[inline]
    // fn wrap(&self) {}
}

impl<T, S: Shape> DevicelessAble<'_, T, S> for CPU<Base> {}

impl<Mods> HasModules for CPU<Mods> {
    type Mods = Mods;
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

impl CPU {
    #[inline]
    pub fn based() -> CPU<Base> {
        CPU::<Base>::new()
    }
}

impl<Mods> CPU<Mods> {
    #[inline]
    pub fn add_layer<Mod>(self) -> CPU<Mod::Wrapped>
    where
        Mod: AddLayer<Mods, CPU>,
    {
        CPU {
            modules: Mod::wrap_layer(self.modules),
        }
    }

    pub fn remove_layer<NewMods>(self) -> CPU<NewMods>
    where
        Mods: RemoveLayer<NewMods>,
    {
        CPU {
            modules: self.modules.inner_mods(),
        }
    }
}

impl<T, Mods: OnDropBuffer> Alloc<T> for CPU<Mods> {
    fn alloc<S: Shape>(&self, mut len: usize, flag: AllocFlag) -> crate::Result<Self::Base<T, S>> {
        assert!(len > 0, "invalid buffer len: 0");

        if S::LEN > len {
            len = S::LEN
        }

        // self.wrap_in_base(CPUPtr::new_initialized(len, flag))
        Ok(CPUPtr::new_initialized(len, flag))
    }

    fn alloc_from_slice<S>(&self, data: &[T]) -> crate::Result<Self::Base<T, S>>
    where
        S: Shape,
        T: Clone,
    {
        assert!(!data.is_empty(), "invalid buffer len: 0");
        assert!(S::LEN <= data.len(), "invalid buffer len: {}", data.len());

        let cpu_ptr = unsafe { CPUPtr::new(data.len(), AllocFlag::None) };
        let slice = unsafe { std::slice::from_raw_parts_mut(cpu_ptr.ptr, data.len()) };
        slice.clone_from_slice(data);

        Ok(cpu_ptr)
    }

    fn alloc_from_vec<S: Shape>(&self, mut vec: Vec<T>) -> crate::Result<Self::Base<T, S>>
    where
        T: Clone,
    {
        assert!(!vec.is_empty(), "invalid buffer len: 0");

        let ptr = vec.as_mut_ptr();
        let len = vec.len();
        core::mem::forget(vec);

        Ok(unsafe { CPUPtr::from_ptr(ptr, len, AllocFlag::None) })
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
{
    #[inline]
    fn clone_buf(&'a self, buf: &Buffer<'a, T, CPU<Mods>, S>) -> Buffer<'a, T, CPU<Mods>, S> {
        let mut cloned = Buffer::new(self, buf.len());
        cloned.clone_from_slice(buf);
        cloned
    }
}

impl<Mods: OnDropBuffer + 'static> UnaryFusing for CPU<Mods> {
    #[cfg(feature = "lazy")]
    #[cfg(feature = "graph")]
    #[inline]
    fn unary_fuse_op<T: Copy + 'static>(
        &self,
    ) -> fn(
        &mut (
            &mut Buffer<'_, T, Self, ()>,
            &Buffer<'_, T, Self, ()>,
            crate::NoId<Vec<std::rc::Rc<dyn Fn(crate::Resolve<T>) -> Box<dyn crate::TwoWay<T>>>>>,
        ),
    ) -> crate::Result<()> {
        |(out, buf, ops)| {
            for (out, buf) in out.iter_mut().zip(buf.iter()) {
                let mut current_val = *buf;
                for op in ops.iter() {
                    let resolve = Resolve {
                        val: current_val,
                        marker: "x",
                    };
                    current_val = op(resolve).eval();
                }
                *out = current_val;
            }
            Ok(())
        }
    }
}

unsafe impl<Mods: OnDropBuffer> IsShapeIndep for CPU<Mods> {}

#[cfg(test)]
mod tests {
    #[cfg(feature = "fork")]
    #[cfg(feature = "cached")]
    #[test]
    fn test_add_layer_cpu() {
        use crate::{Base, CPU};

        let cpu = CPU::<Base>::new();
        let cpu = cpu.add_layer::<crate::Cached<()>>();
        let cpu = cpu.add_layer::<crate::Fork<()>>();

        let _cpu = cpu.remove_layer();
    }
}
