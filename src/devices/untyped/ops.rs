use core::mem::ManuallyDrop;

use crate::{
    cpu::CPUPtr, cpu_stack_ops::apply_fn_slice, untyped::untyped_device::UntypedDevice,
    ApplyFunction, Buffer, Retrieve, Retriever, Shape, CPU,
};

use super::{untyped_device::Untyped, AsType, MatchesType};

impl<T, S> ApplyFunction<T, S> for Untyped
where
    T: Copy + AsType,
    S: Shape,
{
    fn apply_fn<F>(
        &self,
        // buf: &D::Data<T, S>,
        buf: &crate::Buffer<T, Self, S>,
        f: impl Fn(crate::Resolve<T>) -> F + Copy + 'static,
    ) -> crate::Buffer<T, Self, S>
    where
        F: crate::TwoWay<T> + 'static,
    {
        match &self.device {
            UntypedDevice::CPU(cpu) => {
                // let mut out = cpu.retrieve(buf.len(), buf);
                let x = buf.base().convert_to_typed::<T, CPU, S>().unwrap();
                // apply_fn_slice(x, &mut out, f);
                // let mut out = ManuallyDrop::new(out);
                // let data = std::mem::take(out.base_mut());
                // out.
            }
            UntypedDevice::CUDA(cuda) => todo!(),
        }
        todo!()
    }
}
