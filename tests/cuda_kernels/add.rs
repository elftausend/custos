
#[cfg(feature="cuda")]
#[test]
fn test_add_cuda() -> custos::Result<()> {
    use std::ffi::c_void;

    use custos::{CudaDevice, Buffer, cuda::api::{load_module, module_get_fn, launch_kernel}};

    let device = CudaDevice::new(0)?;

    let a = Buffer::from((&device, [1, 2, 3, 4, 5,]));
    let b = Buffer::from((&device, [4, 1, 7, 6, 9,]));

    let module = load_module("tests/cuda_kernels/add.ptx")?;
    let function = module_get_fn(module, "add")?;

    launch_kernel(
        function, [a.len as u32, 1, 1], 
        [1, 1, 1], &mut [a.ptr.2 as *mut c_void, b.ptr.2 as *mut c_void]
    );

    Ok(())
}