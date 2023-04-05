//! CUDA API

#[cfg(test)]
mod test_occupancy;

mod cuda;
mod error;
mod ffi;

pub(crate) mod cublas;
pub(crate) mod nvrtc;

pub use cuda::*;
pub use ffi::*;

#[cfg(test)]
mod tests {
    use crate::cuda::api::CUdeviceptr;

    #[test]
    fn test_cuda_alloc() {
        use crate::cuda::api::{create_context, cuInit, cumalloc, device};

        unsafe { cuInit(0) };
        let device = device(0).unwrap();
        let _ctx = create_context(&device).unwrap();

        let _x = cumalloc::<f32>(10).unwrap();
    }

    #[test]
    fn test_cuda_alloc2() -> crate::Result<()> {
        use crate::cuda::api::{create_context, cuInit, cumalloc, device, device_count};

        unsafe { cuInit(0) };
        println!("count: {}", device_count()?);

        let device = device(0)?;
        let _ctx = create_context(&device)?;

        let _x = cumalloc::<f32>(10)?;

        Ok(())
    }

    #[test]
    fn test_cuda_write() -> crate::Result<()> {
        use crate::cuda::api::{create_context, cuInit, cu_read, cu_write, cumalloc, device};

        unsafe { cuInit(0) };

        let device = device(0)?;
        let _ctx = create_context(&device)?;

        let x = cumalloc::<f32>(100)?;

        let write = [4f32; 10];
        cu_write(x, &write)?;

        let mut read = vec![0f32; 10];
        cu_read(&mut read, x)?;

        assert_eq!(&[4.0; 10], read.as_slice());

        Ok(())
    }

    #[test]
    fn test_cublas() -> crate::Result<()> {
        use crate::cuda::api::{
            create_context, cuInit, cu_read, cu_write,
            cublas::{cublasContext, cublasCreate_v2, cublasOperation_t, cublasSgemm_v2},
            cumalloc, device,
        };
        use std::ptr::null_mut;

        let m = 3;
        let k = 2;
        let n = 3;

        unsafe { cuInit(0) };

        let device = device(0)?;
        let _ctx = create_context(&device)?;

        let a = cumalloc::<f32>(m * k)?;

        let write = (0..m * k).map(|x| x as f32).collect::<Vec<f32>>();
        cu_write(a, &write)?;

        let b = cumalloc::<f32>(k * n)?;

        let write = (0..k * n).rev().map(|x| x as f32).collect::<Vec<f32>>();
        cu_write(b, &write)?;

        let c = cumalloc::<f32>(m * n)?;

        unsafe {
            let mut handle: *mut cublasContext = null_mut();
            let res = cublasCreate_v2(&mut handle);
            if res as u32 != 0 {
                println!("cublas create")
            }

            let res = cublasSgemm_v2(
                handle,
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &1f32 as *const f32,
                b as *const u64 as *const f32,
                n as i32,
                a as *const u64 as *const f32,
                k as i32,
                &0f32 as *const f32,
                c as *mut u64 as *mut f32,
                n as i32,
            );
            if res as u32 != 0 {
                println!("cublas gemm")
            }
            let mut read = vec![0f32; n * m];
            cu_read(&mut read, c)?;
            println!("read: {read:?}");
        }
        Ok(())
    }

    const N: usize = 100;

    #[test]
    fn test_ffi_cuda() {
        use std::{ffi::c_void, mem::size_of};

        use crate::cuda::api::{
            cuCtxCreate_v2, cuDeviceGet, cuInit, cuMemAlloc_v2, cuMemcpyDtoH_v2, cuMemcpyHtoD_v2,
            CUctx_st,
        };

        unsafe {
            let mut device = 0;
            let mut context: *mut CUctx_st = std::ptr::null_mut();

            let a: Vec<f32> = (0..N).into_iter().map(|x| x as f32).collect();
            let mut a_d: CUdeviceptr = 0;

            let mut out = [0f32; N];

            cuInit(0).to_result().unwrap();
            cuDeviceGet(&mut device, 0).to_result().unwrap();
            cuCtxCreate_v2(&mut context, 0, device).to_result().unwrap();

            cuMemAlloc_v2(&mut a_d, N * size_of::<f32>());

            cuMemcpyHtoD_v2(a_d, a.as_ptr() as *const c_void, N * size_of::<f32>())
                .to_result()
                .unwrap();
            cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, a_d, N * size_of::<f32>())
                .to_result()
                .unwrap();
            println!("out: {out:?}");
        };
    }

    #[test]
    fn test_cuda_device() -> crate::Result<()> {
        use crate::{cuda::CUDA, Buffer};

        let device = CUDA::new(0)?;
        let _a = Buffer::<f32, _>::new(&device, 10);
        Ok(())
    }
}
