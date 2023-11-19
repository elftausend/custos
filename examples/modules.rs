use std::ops::{Add, AddAssign, Deref, DerefMut, Mul};

use custos::{
    AddGradFn, AddOperation, Alloc, Buffer, Device, MayTapeActions, Retrieve, Retriever, Shape,
    UseGpuOrCpu, CPU,
};

pub trait ElementWise<T, D: Device, S: Shape>: Device {
    fn add(
        &self,
        lhs: &Buffer<T, D, S>,
        rhs: &Buffer<T, D, S>,
    ) -> custos::Result<Buffer<T, Self, S>>;
}

pub fn add_ew_slice<T: Add<Output = T> + Copy>(lhs: &[T], rhs: &[T], out: &mut [T]) {
    for ((lhs, rhs), out) in lhs.iter().zip(rhs).zip(out) {
        *out = *lhs + *rhs;
    }
}

pub fn add_ew_grad_slice<T>(lhs_grad: &mut [T], rhs_grad: &mut [T], out: &[T])
where
    T: Copy + AddAssign + Mul<Output = T>,
{
    for ((lhs_grad, rhs_grad), out) in lhs_grad.iter_mut().zip(rhs_grad).zip(out) {
        *lhs_grad += *out;
        *rhs_grad += *out;
    }
}

impl<T, D, S, Mods> ElementWise<T, D, S> for CPU<Mods>
where
    T: Add<Output = T> + AddAssign + Mul<Output = T> + Copy + 'static,
    D: Device + Alloc<T> + MayTapeActions + 'static,
    D::Data<T, S>: Deref<Target = [T]> + DerefMut,
    S: Shape,
    Mods: Retrieve<Self, T> + AddOperation + MayTapeActions + AddGradFn + 'static,
{
    #[track_caller]
    fn add(
        &self,
        lhs: &Buffer<T, D, S>,
        rhs: &Buffer<T, D, S>,
    ) -> custos::Result<Buffer<T, Self, S>> {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs));

        self.add_grad_fn((lhs, rhs, &mut out), |(lhs, rhs, out)| {
            add_ew_grad_slice(lhs.grad_mut(), rhs.grad_mut(), out.grad()); // execute grad function
            Ok(())
        });

        self.add_op((lhs, rhs, &mut out), |(lhs, rhs, out)| {
            add_ew_slice(lhs, rhs, out);
            Ok(())
        })?;
        Ok(out)
    }
}

#[cfg(feature = "opencl")]
use custos::{opencl::CLPtr, CDatatype, OpenCL};

#[cfg(feature = "opencl")]
pub fn try_add_ew_cl<T, Mods>(
    device: &OpenCL<Mods>,
    lhs: &CLPtr<T>,
    rhs: &CLPtr<T>,
    out: &mut CLPtr<T>,
) -> custos::Result<()>
where
    T: CDatatype + Default,
{
    let src = format!(
        "
        __kernel void add_ew(__global const {ty}* lhs, __global const {ty}* rhs, __global {ty}* out) {{
            size_t id = get_global_id(0);
            out[id] = lhs[id] + rhs[id];
        }}
    ",
        ty = T::C_DTYPE_STR,
    );

    device.launch_kernel(
        &src,
        [((lhs.len + 32) / 32) * 32, 0, 0],
        Some([32, 0, 0]),
        &[lhs, rhs, out],
    )
}

#[cfg(feature = "opencl")]
impl<T, S, Mods> ElementWise<T, Self, S> for custos::OpenCL<Mods>
where
    T: Add<Output = T> + Copy + CDatatype + Default,
    S: Shape,
    Mods: Retrieve<Self, T> + AddOperation + UseGpuOrCpu + 'static,
{
    fn add(
        &self,
        lhs: &Buffer<T, Self, S>,
        rhs: &Buffer<T, Self, S>,
    ) -> custos::Result<Buffer<T, Self, S>> {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs));

        self.add_op((lhs, rhs, &mut out), |(lhs, rhs, out)| {
            let dev = lhs.device();
            let out = &mut **out;
            #[cfg(unified_cl)]
            {
                let cpu_out = unsafe { &mut *(out as *mut Buffer<_, OpenCL<Mods>, _>) };
                dev.use_cpu_or_gpu(
                    (file!(), line!(), column!()).into(),
                    &[lhs.len()],
                    || add_ew_slice(lhs, rhs, cpu_out),
                    || try_add_ew_cl(dev, &lhs.data, &rhs.data, &mut out.data).unwrap(),
                );
            }
            // #[cfg(not(unified_cl))]
            try_add_ew_cl(dev, &lhs.data, &rhs.data, &mut out.data)?;
            Ok(())
        })?;

        Ok(out)
    }
}

fn main() {
    #[cfg(feature = "opencl")]
    {
        use custos::{Base, Cached, Fork, Lazy, Run};
        let device = OpenCL::<Fork<Lazy<Cached<Base>>>>::new(0).unwrap();
        let lhs = device.buffer([1, 2, 3, 4, 5]);
        let rhs = device.buffer([1, 2, 3, 4, 5]);

        let out = device.add(&lhs, &rhs).unwrap();
        unsafe { device.run().unwrap() };
        assert_eq!(out.read(), vec![2, 4, 6, 8, 10])
    }
}
