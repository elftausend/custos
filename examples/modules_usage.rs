use std::ops::{Add, AddAssign, Deref, DerefMut, Mul};

use custos::{
    AddGradFn, AddOperation, Alloc, ApplyFunction, Buffer, Combiner, Device, MayGradActions,
    Retrieve, Retriever, Shape, Unit, UseGpuOrCpu, ZeroGrad, CPU,
};

pub trait ElementWise<T: Unit, D: Device, S: Shape>: Device {
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
    T: Unit + Add<Output = T> + AddAssign + Mul<Output = T> + Default + Copy + 'static,
    D: Device + ZeroGrad<T> + Alloc<T> + MayGradActions + 'static,
    D::Base<T, S>: Deref<Target = [T]> + DerefMut,
    S: Shape,
    Mods: Retrieve<Self, T, S> + AddOperation + MayGradActions + AddGradFn + 'static,
{
    fn add(
        &self,
        lhs: &Buffer<T, D, S>,
        rhs: &Buffer<T, D, S>,
    ) -> custos::Result<Buffer<T, Self, S>> {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs)).unwrap();

        self.add_grad_fn((lhs, rhs, &mut out), |(lhs, rhs, out)| unsafe {
            add_ew_grad_slice(lhs.grad_mut_unbound(), rhs.grad_mut_unbound(), out.grad()); // execute grad function
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
    Mods: Retrieve<Self, T, S> + AddOperation + UseGpuOrCpu + 'static,
{
    fn add(
        &self,
        lhs: &Buffer<T, Self, S>,
        rhs: &Buffer<T, Self, S>,
    ) -> custos::Result<Buffer<T, Self, S>> {
        let mut out = self.retrieve(lhs.len(), (lhs, rhs)).unwrap();

        self.add_op((lhs, rhs, &mut out), |(lhs, rhs, out)| {
            let dev = lhs.device();
            #[cfg(unified_cl)]
            {
                let cpu_out = unsafe { &mut *(out as *mut Buffer<_, OpenCL<Mods>, _>) };
                dev.use_cpu_or_gpu(
                    (file!(), line!(), column!()).into(),
                    &[lhs.len()],
                    || add_ew_slice(lhs, rhs, cpu_out),
                    || try_add_ew_cl(dev, lhs, rhs, out).unwrap(),
                );
            }
            #[cfg(not(unified_cl))]
            try_add_ew_cl(dev, lhs, rhs, out)?;
            Ok(())
        })?;

        Ok(out)
    }
}

fn main() {
    // only the Base module
    #[cfg(feature = "cpu")]
    {
        use custos::Base;
        let device = CPU::<Base>::new();
        let lhs = Buffer::from((&device, &[1, 2, 3, 4, 5]));
        let rhs = Buffer::from((&device, &[1, 2, 3, 4, 5]));

        let out = device.add(&lhs, &rhs).unwrap();
        assert_eq!(out.read(), [2, 4, 6, 8, 10])
    }

    // lazy module
    #[cfg(feature = "cpu")]
    #[cfg(feature = "lazy")]
    {
        use custos::{Base, Lazy, Run};
        let device = CPU::<Lazy<Base>>::new();
        let lhs = Buffer::from((&device, &[1, 2, 3, 4, 5]));
        // this works too
        let rhs = device.buffer([1, 2, 3, 4, 5]);

        let out = device.add(&lhs, &rhs).unwrap();
        unsafe { device.run().unwrap() }; // allocates memory and executes all operations inside the lazy graph
        assert_eq!(out.replace().read(), [2, 4, 6, 8, 10])
    }

    // lazy & graph modules
    #[cfg(feature = "opencl")]
    #[cfg(feature = "lazy")]
    #[cfg(feature = "graph")]
    {
        use custos::{Base, Graph, Lazy, Optimize, Run};
        let device = OpenCL::<Graph<Lazy<Base>>>::new(0).unwrap();
        // should work with any device (except nnapi)
        // let device = CPU::<Graph<Lazy<Base>>>::new();
        let buf = device.buffer([1., 2., 3., 4., 5.]);

        let out1 = device.apply_fn(&buf, |x| x.add(1.));
        let out2 = device.apply_fn(&out1, |x| x.sin());

        // this identifies redundant intermediate buffers and skips allocating them
        device.optimize_mem_graph(&device, None).unwrap(); // allocates, now out1 data points to out2 data. The data is accessed with out2.replace()
                                                           // this fuses all unary operations and creates fused compute kernels (for all compute kernel based devices)
        device.unary_fusing(&device, None).unwrap();

        // this executes all operations inside the lazy graph
        unsafe { device.run().unwrap() };

        for (input, out) in buf.read().iter().zip(out2.replace().read()) {
            assert!((out - (input + 1.).sin()).abs() < 0.01);
        }
    }

    // fork
    #[cfg(feature = "fork")]
    #[cfg(feature = "lazy")]
    #[cfg(feature = "opencl")]
    {
        use custos::{Base, Fork, Lazy, Run};
        let device = OpenCL::<Fork<Lazy<Base>>>::new(0).unwrap();
        let lhs = device.buffer([1, 2, 3, 4, 5]);
        let rhs = device.buffer([1, 2, 3, 4, 5]);

        let out = device.add(&lhs, &rhs).unwrap();
        unsafe { device.run().unwrap() };
        assert_eq!(out.replace().read(), vec![2, 4, 6, 8, 10])
    }
}
