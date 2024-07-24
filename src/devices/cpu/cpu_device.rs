use core::convert::Infallible;

use crate::{
    cpu::CPUPtr, flag::AllocFlag, impl_device_traits, AddLayer, Alloc, Base, Buffer, CloneBuf,
    Device, DeviceError, DevicelessAble, HasModules, IsShapeIndep, Module, OnDropBuffer,
    OnNewBuffer, RemoveLayer, Setup, Shape, UnaryFusing, Unit, WrappedData,
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
    type Base<T: Unit, S: Shape> = CPUPtr<T>;
    type Data<T: Unit, S: Shape> = Self::Wrap<T, Self::Base<T, S>>;
    // type WrappedData<T, S: Shape> = ;

    fn new() -> Result<Self, Self::Error> {
        todo!()
    }

    #[inline(always)]
    fn base_to_data<T: Unit, S: Shape>(&self, base: Self::Base<T, S>) -> Self::Data<T, S> {
        self.wrap_in_base(base)
    }

    #[inline(always)]
    fn wrap_to_data<T: Unit, S: Shape>(
        &self,
        wrap: Self::Wrap<T, Self::Base<T, S>>,
    ) -> Self::Data<T, S> {
        wrap
    }

    #[inline(always)]
    fn data_as_wrap<T: Unit, S: Shape>(
        data: &Self::Data<T, S>,
    ) -> &Self::Wrap<T, Self::Base<T, S>> {
        data
    }

    #[inline(always)]
    fn data_as_wrap_mut<T: Unit, S: Shape>(
        data: &mut Self::Data<T, S>,
    ) -> &mut Self::Wrap<T, Self::Base<T, S>> {
        data
    }

    // #[inline]
    // fn wrap(&self) {}
}

impl<T: Unit, S: Shape> DevicelessAble<'_, T, S> for CPU<Base> {}

impl<Mods> HasModules for CPU<Mods> {
    type Mods = Mods;
    #[inline]
    fn modules(&self) -> &Mods {
        &self.modules
    }
}

impl<SimpleMods> CPU<SimpleMods> {
    #[inline]
    pub fn new<'a, NewMods>() -> CPU<SimpleMods::Module>
    where
        Self: 'a,
        SimpleMods: Module<'a, CPU, Module = NewMods>,
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

impl<T: Unit, Mods: OnDropBuffer> Alloc<T> for CPU<Mods> {
    fn alloc<S: Shape>(&self, mut len: usize, flag: AllocFlag) -> crate::Result<Self::Base<T, S>> {
        if len == 0 {
            return Err(DeviceError::ZeroLengthBuffer.into());
        }

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
        if data.is_empty() {
            return Err(DeviceError::ZeroLengthBuffer.into());
        }
        if !(S::LEN == data.len() || S::LEN == 0) {
            return Err(DeviceError::ShapeLengthMismatch.into());
        }

        let cpu_ptr = unsafe { CPUPtr::new(data.len(), AllocFlag::None) };
        let slice = unsafe { std::slice::from_raw_parts_mut(cpu_ptr.ptr, data.len()) };
        slice.clone_from_slice(data);

        Ok(cpu_ptr)
    }

    fn alloc_from_vec<S: Shape>(&self, vec: Vec<T>) -> crate::Result<Self::Base<T, S>>
    where
        T: Clone,
    {
        if vec.is_empty() {
            return Err(DeviceError::ZeroLengthBuffer.into());
        }
        Ok(CPUPtr::from_vec(vec))
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

impl<'a, Mods: OnDropBuffer + OnNewBuffer<'a, T, Self, S>, T: Unit + Clone, S: Shape>
    CloneBuf<'a, T, S> for CPU<Mods>
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
    fn unary_fuse_op<T: Unit + Copy + 'static>(
        &self,
        ops_to_fuse: Vec<std::rc::Rc<dyn Fn(crate::Resolve<T>) -> Box<dyn crate::TwoWay<T>>>>,
    ) -> Box<dyn Fn((&mut Buffer<'_, T, Self, ()>, &Buffer<'_, T, Self, ()>)) -> crate::Result<()>>
    {
        Box::new(move |(out, buf)| {
            for (out, buf) in out.iter_mut().zip(buf.iter()) {
                let mut current_val = *buf;
                for op in ops_to_fuse.iter() {
                    let resolve = crate::Resolve {
                        val: current_val,
                        marker: "x",
                    };
                    current_val = op(resolve).eval();
                }
                *out = current_val;
            }
            Ok(())
        })
    }
}

unsafe impl<Mods: OnDropBuffer> IsShapeIndep for CPU<Mods> {}

#[cfg(test)]
mod tests {
    use crate::{Alloc, DeviceError, Dim1, CPU};

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

    #[test]
    fn test_alloc_shape_size_mismatch_cpu() {
        let device = CPU::based();
        let res = device.alloc_from_slice::<Dim1<3>>(&[1, 2, 3, 4]);
        if let Err(e) = res {
            let e = e.downcast_ref::<DeviceError>().unwrap();
            if e != &DeviceError::ShapeLengthMismatch {
                panic!()
            }
        } else {
            panic!()
        }
        device.alloc_from_slice::<()>(&[1, 2, 3, 4]).unwrap();
    }

    #[test]
    fn test_size_zero_alloc_cpu() {
        let device = CPU::based();
        let res = Alloc::<i32>::alloc_from_slice::<()>(&device, &[]);
        if let Err(e) = res {
            let e = e.downcast_ref::<DeviceError>().unwrap();
            if e != &DeviceError::ZeroLengthBuffer {
                panic!()
            }
        } else {
            panic!()
        }

        let res = Alloc::<i32>::alloc::<()>(&device, 0, crate::flag::AllocFlag::None);
        if let Err(e) = res {
            let e = e.downcast_ref::<DeviceError>().unwrap();
            if e != &DeviceError::ZeroLengthBuffer {
                panic!()
            }
        } else {
            panic!()
        }

        let res = Alloc::<i32>::alloc_from_vec::<()>(&device, vec![]);
        if let Err(e) = res {
            let e = e.downcast_ref::<DeviceError>().unwrap();
            if e != &DeviceError::ZeroLengthBuffer {
                panic!()
            }
        } else {
            panic!()
        }
    }
}
