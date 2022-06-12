
#[cfg(feature="cuda")]
#[test]
fn test_cuda_alloc() {
    use custos::cuda::api::{cumalloc, cuInit, device, create_context};

    unsafe { cuInit(0) };
    let device = device(0).unwrap();
    let _ctx = create_context(device).unwrap();

    let _x = cumalloc::<f32>(10).unwrap();

}
#[cfg(feature="cuda")]
#[test]
fn test_cuda_alloc2() -> custos::Result<()> {
    use custos::cuda::api::{cumalloc, cuInit, device, create_context, device_count};

    unsafe { cuInit(0) };
    println!("count: {}", device_count()?);

    let device = device(1)?;
    let _ctx = create_context(device)?;

    let _x = cumalloc::<f32>(10)?;

    Ok(())
}

#[cfg(feature="cuda")]
#[test]
fn test_cuda_write() -> custos::Result<()> {
    use custos::cuda::api::{cumalloc, cuInit, device, create_context, cuwrite, curead};

    unsafe { cuInit(0) };

    let device = device(0)?;
    let _ctx = create_context(device)?;

    let x = cumalloc::<f32>(100)?;
    
    let write = [4f32; 10];
    cuwrite(x, &write)?;    

    let mut read = vec![0f32; 10];
    curead(&mut read, x)?;
    
    assert_eq!(&[4.0; 10], read.as_slice());

    Ok(())
}

const N: usize = 100;

#[cfg(feature="cuda")]
#[test]
fn test_ffi_cuda() {
    use std::{ffi::c_void, mem::size_of};

    use custos::cuda::api::{cuInit, cuDeviceGet, cuCtxCreate_v2, CUctx_st, cuMemAlloc_v2, CUdeviceptr, cuMemcpyHtoD_v2, cuMemcpyDtoH_v2};

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

        cuMemcpyHtoD_v2(a_d, a.as_ptr() as *const c_void, N * size_of::<f32>()).to_result().unwrap();
        cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, a_d, N * size_of::<f32>()).to_result().unwrap();
        println!("out: {out:?}");
    };
    

}