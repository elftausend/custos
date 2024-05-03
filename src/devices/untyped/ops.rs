use crate::{
    cpu_stack_ops::apply_fn_slice, untyped::untyped_device::UntypedDevice, ApplyFunction,
    CDatatype, Read, Retriever, Shape, CPU,
};

use super::{untyped_device::Untyped, AsType};

impl<T: 'static + AsType + Default + Clone, S: Shape> Read<T, S> for Untyped {
    type Read<'a> = Vec<T>
    where
        T: 'a,
        Self: 'a,
        S: 'a;

    #[inline]
    fn read<'a>(&self, buf: &'a Self::Base<T, S>) -> Self::Read<'a>
    where
        Self: 'a,
    {
        Read::<T, S>::read_to_vec(self, buf)
    }

    fn read_to_vec(&self, buf: &Self::Base<T, S>) -> Vec<T>
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
        cpu::CPUPtr,
        tests_helper::{add_ew_slice, roughly_eq_slices, AddEw},
        untyped::{
            storages::{CpuStorage, CudaStorage, UntypedData},
            untyped_device::{Untyped, UntypedDevice},
        },
        ApplyFunction, Buffer, Combiner, Device, Shape,
    };

    #[test]
    fn test_apply_fn_untyped() {
        let device = Untyped::new().unwrap();
        let res = device.buffer([1., 2., 3., 4.]);
        let out = device.apply_fn(&res, |x| x.add(1.));
        roughly_eq_slices(&out.read(), &[2., 3., 4., 5.]);
    }

    fn alloc_and_add_slice<T: Copy + std::ops::Add<Output = T>>(lhs: &[T], rhs: &[T]) -> CPUPtr<T> {
        let mut out = unsafe { CPUPtr::<T>::new(lhs.len(), crate::flag::AllocFlag::None) };
        add_ew_slice(lhs, rhs, &mut out);
        out
    }

    #[cfg(feature = "cuda")]
    fn alloc_and_add_cu<T: crate::CDatatype>(
        device: &crate::cuda::CudaDevice,
        lhs: &crate::cuda::CUDAPtr<T>,
        rhs: &crate::cuda::CUDAPtr<T>,
    ) -> crate::cuda::CUDAPtr<T> {
        let mut out = crate::cuda::CUDAPtr::new(lhs.len, crate::flag::AllocFlag::None).unwrap();
        let src = format!(
            r#"extern "C" __global__ void addEw({dt}* lhs, {dt}* rhs, {dt}* out, int len) {{
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= len) {{
                    return;
                }}
                out[idx] = lhs[idx] + rhs[idx];
            }}    
        "#,
            dt = T::C_DTYPE_STR
        );

        device
            .launch_kernel(
                src,
                "addEw",
                [(lhs.len / 32) as u32 + 1, 1, 1],
                [32, 1, 1],
                0,
                &[lhs, rhs, &mut out, &lhs.len],
            )
            .unwrap();
        out
    }

    impl<T, S: Shape> AddEw<T, Self, S> for Untyped {
        fn add(&self, lhs: &Buffer<T, Self, S>, rhs: &Buffer<T, Self, S>) -> Buffer<T, Self, S> {
            let out = match (&lhs.data, &rhs.data) {
                (UntypedData::CPU(lhs), UntypedData::CPU(rhs)) => {
                    let storage = match (lhs, rhs) {
                        (CpuStorage::U8(lhs), CpuStorage::U8(rhs)) => {
                            CpuStorage::U8(alloc_and_add_slice(lhs, rhs))
                        }
                        (CpuStorage::U32(lhs), CpuStorage::U32(rhs)) => {
                            CpuStorage::U32(alloc_and_add_slice(lhs, rhs))
                        }
                        (CpuStorage::I64(lhs), CpuStorage::I64(rhs)) => {
                            CpuStorage::I64(alloc_and_add_slice(lhs, rhs))
                        }
                        #[cfg(feature = "half")]
                        (CpuStorage::BF16(lhs), CpuStorage::BF16(rhs)) => {
                            CpuStorage::BF16(alloc_and_add_slice(lhs, rhs))
                        }
                        #[cfg(feature = "half")]
                        (CpuStorage::F16(lhs), CpuStorage::F16(rhs)) => {
                            CpuStorage::F16(alloc_and_add_slice(lhs, rhs))
                        }
                        (CpuStorage::F32(lhs), CpuStorage::F32(rhs)) => {
                            CpuStorage::F32(alloc_and_add_slice(lhs, rhs))
                        }
                        (CpuStorage::F64(lhs), CpuStorage::F64(rhs)) => {
                            CpuStorage::F64(alloc_and_add_slice(lhs, rhs))
                        }
                        _ => unimplemented!(),
                    };
                    UntypedData::CPU(storage)
                }
                (UntypedData::CUDA(lhs), UntypedData::CUDA(rhs)) => {
                    let UntypedDevice::CUDA(dev) = &self.device else {
                        panic!()
                    };
                    #[cfg(feature = "cuda")]
                    let storage = match (lhs, rhs) {
                        (CudaStorage::U8(lhs), CudaStorage::U8(rhs)) => {
                            CudaStorage::U8(alloc_and_add_cu(dev, lhs, rhs))
                        }
                        (CudaStorage::U32(lhs), CudaStorage::U32(rhs)) => {
                            CudaStorage::U32(alloc_and_add_cu(dev, lhs, rhs))
                        }
                        (CudaStorage::I64(lhs), CudaStorage::I64(rhs)) => {
                            CudaStorage::I64(alloc_and_add_cu(dev, lhs, rhs))
                        }
                        #[cfg(feature = "half")]
                        (CudaStorage::BF16(lhs), CudaStorage::BF16(rhs)) => {
                            CudaStorage::BF16(alloc_and_add_cu(dev, lhs, rhs))
                        }
                        #[cfg(feature = "half")]
                        (CudaStorage::F16(lhs), CudaStorage::F16(rhs)) => {
                            CudaStorage::F16(alloc_and_add_cu(dev, lhs, rhs))
                        }
                        (CudaStorage::F32(lhs), CudaStorage::F32(rhs)) => {
                            CudaStorage::F32(alloc_and_add_cu(dev, lhs, rhs))
                        }
                        #[cfg(not(target_os = "macos"))]
                        (CudaStorage::F64(lhs), CudaStorage::F64(rhs)) => {
                            CudaStorage::F64(alloc_and_add_cu(dev, lhs, rhs))
                        }
                        _ => unimplemented!(),
                    };

                    #[cfg(feature = "cuda")]
                    {
                        UntypedData::CUDA(storage)
                    }
                    #[cfg(not(feature = "cuda"))]
                    unimplemented!()
                }
                _ => unimplemented!(),
            };
            Buffer {
                data: out,
                device: Some(self),
            }
        }
    }

    #[test]
    fn test_untyped_ew_add() {
        let device = Untyped::new().unwrap();
        let lhs = device.buffer([1f32, 2., 3., 4.]).to_untyped();
        let rhs = device.buffer([1f32, 2., 3., 4.]).to_untyped();

        let out = device.add(&lhs, &rhs);
        roughly_eq_slices(&[2., 4., 6., 8.], &out.read_typed::<f32>())
    }
}
