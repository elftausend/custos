use custos::{opencl::enqueue_kernel, Buffer, CDatatype, OnDropBuffer, OpenCL, Retriever, Shape};

use super::ElementWise;

pub fn cl_element_wise<Mods: OnDropBuffer, T, S: Shape>(
    device: &OpenCL<Mods>,
    lhs: &Buffer<T, OpenCL<Mods>, S>,
    rhs: &Buffer<T, OpenCL<Mods>, S>,
    out: &mut Buffer<T, OpenCL<Mods>, S>,
    op: &str,
) -> custos::Result<()>
where
    T: CDatatype,
{
    let src = format!(
        "__kernel void cl_ew(__global {datatype}* lhs, __global {datatype}* rhs, __global {datatype}* out) {{
            size_t idx = get_global_id(0);

            out[idx] = lhs[idx] {op} rhs[idx];
        }}"
    , datatype=T::C_DTYPE_STR);

    enqueue_kernel(device, &src, [lhs.len(), 0, 0], None, &[lhs, rhs, out])?;
    Ok(())
}

impl<T: CDatatype, S: Shape> ElementWise<T, OpenCL, S> for OpenCL {
    #[inline]
    fn add(&self, lhs: &Buffer<T, OpenCL, S>, rhs: &Buffer<T, OpenCL, S>) -> Buffer<T, OpenCL, S> {
        let mut out = self.retrieve(lhs.len(), ());
        cl_element_wise(self, lhs, rhs, &mut out, "+").unwrap();
        out
    }

    #[inline]
    fn mul(&self, lhs: &Buffer<T, OpenCL, S>, rhs: &Buffer<T, OpenCL, S>) -> Buffer<T, OpenCL, S> {
        let mut out = self.retrieve(lhs.len(), ());
        cl_element_wise(self, lhs, rhs, &mut out, "*").unwrap();
        out
    }
}

#[cfg(test)]
mod tests {
    use custos::{prelude::chosen_cl_idx, Base, Buffer, Cached, OpenCL, Retriever, WithShape, CPU};

    use crate::demo_impl::cpu::cpu_element_wise;

    #[test]
    fn test_cl_element_wise() {
        use super::cl_element_wise;

        let device = OpenCL::<Base>::new(chosen_cl_idx()).unwrap();

        let lhs = Buffer::with(&device, [1, 2, 3, 4]);
        let rhs = Buffer::with(&device, [4, 1, 9, 4]);

        let mut out = device.retrieve(lhs.len(), ());

        cl_element_wise(&device, &lhs, &rhs, &mut out, "+").unwrap();

        assert_eq!(out.read(), &[5, 3, 12, 8]);
    }

    const SIZE: usize = 655360;
    const TIMES: usize = 100;

    #[test]
    fn test_element_wise_large_bufs_cl() {
        use super::cl_element_wise;

        let device = OpenCL::<Base>::new(chosen_cl_idx()).unwrap();

        let lhs = Buffer::from((&device, vec![1.0f32; SIZE]));
        let rhs = Buffer::from((&device, vec![4.0; SIZE]));

        let mut out = device.retrieve::<(), 0>(lhs.len(), ());

        let start = std::time::Instant::now();

        for _ in 0..TIMES {
            cl_element_wise::<_, _, ()>(&device, &lhs, &rhs, &mut out, "+").unwrap();
            // assert_eq!(out.read(), &[5.0; SIZE]);
        }

        println!("ocl: {:?}", start.elapsed() /*/ TIMES as u32*/);

        assert_eq!(out.read(), &[5.0; SIZE]);
    }

    #[test]
    fn test_element_wise_large_bufs_cpu() {
        let device = CPU::<Base>::new();

        let lhs = Buffer::<_>::from((&device, vec![1.0f32; SIZE]));
        let rhs = Buffer::<_>::from((&device, vec![4.0; SIZE]));

        let mut out = device.retrieve::<(), 0>(lhs.len(), ());

        let start = std::time::Instant::now();
        for _ in 0..TIMES {
            cpu_element_wise(&lhs, &rhs, &mut out, |out, lhs, rhs| *out = lhs + rhs);
        }

        println!("cpu: {:?}", start.elapsed());
        assert_eq!(out.as_slice(), &[5.0; SIZE]);
    }
}
