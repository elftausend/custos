use std::ffi::c_void;
use custos::{cuda::api::{nvrtc::create_program, load_module_data, module_get_fn, create_stream, launch_kernel}, CudaDevice, VecRead, Buffer};


#[test]
fn test_nvrtc() -> custos::Result<()> {
    let device = CudaDevice::new(0)?;

    let a = Buffer::from((&device, [1, 2, 3, 4, 5,]));
    let b = Buffer::from((&device, [4, 1, 7, 6, 9,]));

    let c = Buffer::<i32>::new(&device, a.len);

    let src = r#"
        extern "C" __global__ void add(int *a, int *b, int *c)
            {
                int idx = blockIdx.x;
                c[idx] = a[idx] + b[idx];
    }"#;

    let x = create_program(src, "add")?;
    x.compile()?;
    let module = load_module_data(x.ptx()?)?;
    
    let function = module_get_fn(module, "add")?;
    let mut stream = create_stream()?;

    launch_kernel(
        &function, [a.len as u32, 1, 1], 
        [1, 1, 1], &mut stream, &mut [&a.ptr.2 as *const u64 as *mut c_void, &b.ptr.2 as *const u64 as *mut c_void, &c.ptr.2 as *const u64 as *mut c_void]
    )?;

    let read = device.read(&c);
    println!("read: {read:?}");
    Ok(())
}