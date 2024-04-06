use crate::{
    cpu_stack_ops::apply_fn_slice, untyped::untyped_device::UntypedDevice, ApplyFunction,
    CDatatype, Retriever, Shape, CPU,
};

use super::{untyped_device::Untyped, AsType};

impl<T, S> ApplyFunction<T, S> for Untyped
where
    T: CDatatype + Default + Copy + AsType,
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
        let mut out = self.retrieve(buf.len(), buf);
        match &self.device {
            UntypedDevice::CPU(_cpu) => {
                let x = buf.convert_to_typed::<T, CPU, S>().unwrap();
                apply_fn_slice(x, out.convert_to_typed_mut::<T, CPU, S>().unwrap(), f);
            }
            UntypedDevice::CUDA(cuda) => {
                #[cfg(feature = "cuda")]
                {
                    let x = buf.convert_to_typed::<T, crate::CUDA, S>().unwrap();
                    let out = out.convert_to_typed_mut::<T, crate::CUDA, S>().unwrap();
                    crate::cuda::try_cu_apply_fn_mut(&cuda, x, out, f).unwrap();
                }
                #[cfg(not(feature = "cuda"))]
                unimplemented!()
            }
        }
        out
    }
}
