/*
#[test]
fn test_add_cuda() -> custos::Result<()> {
    use std::ffi::c_void;

    use custos::{
        cuda::api::{create_stream, culaunch_kernel, load_module},
        Buffer, CudaDevice, Read,
    };
    let device = CudaDevice::new(0)?;

    let a = Buffer::from((&device, [1, 2, 3, 4, 5]));
    let b = Buffer::from((&device, [4, 1, 7, 6, 9]));

    let c = Buffer::<i32>::new(&device, a.len);

    println!("a: {a:?}");
    println!("b: {b:?}");

    let ptx_path = format!(
        "{}/tests/cuda/cuda_kernels/add.ptx",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    );
    let module = load_module(&ptx_path)?;
    let function = module.function("add")?;
    let mut stream = create_stream()?;

    culaunch_kernel(
        &function,
        [a.len as u32, 1, 1],
        [1, 1, 1],
        &mut stream,
        &mut [
            &a.ptr.2 as *const u64 as *mut c_void,
            &b.ptr.2 as *const u64 as *mut c_void,
            &c.ptr.2 as *const u64 as *mut c_void,
        ],
    )?;

    let read = device.read(&c);
    println!("read: {read:?}");

    Ok(())
}
*/
