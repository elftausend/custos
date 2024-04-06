use crate::{
    cpu_stack_ops::apply_fn_slice, untyped::untyped_device::UntypedDevice, ApplyFunction,
    CDatatype, Read, Retriever, Shape, CPU,
};

use super::{untyped_device::Untyped, AsType};

impl<T: AsType + Default + Clone, S: Shape> Read<T, S> for Untyped {
    type Read<'a> = Vec<T>
    where
        T: 'a,
        Self: 'a,
        S: 'a;

    #[inline]
    fn read<'a>(&self, buf: &'a crate::Buffer<T, Self, S>) -> Self::Read<'a> {
        buf.read_to_vec()
    }

    fn read_to_vec(&self, buf: &crate::Buffer<T, Self, S>) -> Vec<T>
    where
        T: Default + Clone,
    {
        match &self.device {
            UntypedDevice::CPU(_cpu) => buf.convert_to_typed::<T, CPU, S>().unwrap().to_vec(),
            UntypedDevice::CUDA(_cuda) => {
                #[cfg(not(feature = "cuda"))]
                {
                    unimplemented!()
                }

                #[cfg(feature = "cuda")]
                buf.convert_to_typed::<T, super::untyped_device::Cuda<crate::Base>, S>()
                    .unwrap()
                    .read()
            }
        }
    }
}

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

#[cfg(test)]
mod tests {
    use crate::{
        tests_ex::roughly_eq_slices, untyped::untyped_device::Untyped, ApplyFunction, Combiner,
        Device,
    };

    #[test]
    fn test_apply_fn_untyped() {
        let device = Untyped::new().unwrap();
        let res = device.buffer([1., 2., 3., 4.]);
        let out = device.apply_fn(&res, |x| x.add(1.));
        roughly_eq_slices(&out.read(), &[2., 3., 4., 5.]);
    }
}
