use std::{ffi::c_void, time::Instant};

use custos::{
    cuda::api::{
        cuLaunchKernel, cuOccupancyMaxPotentialBlockSize, load_module_data, nvrtc::create_program,
    },
    Buffer, CudaDevice, VecRead,
};

#[test]
fn test_occupancy() -> custos::Result<()> {
    let device = CudaDevice::new(0)?;

    let a = Buffer::from((&device, [1, 2, 3, 4, 5]));
    let b = Buffer::from((&device, [4, 1, 7, 6, 9]));

    let c = Buffer::<i32>::new(&device, a.len);

    let src = r#"
        extern "C" __global__ void add(int *a, int *b, int *c, int numElements)
            {
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx < numElements) {
                    c[idx] = a[idx] + b[idx];
                }
    }"#;

    let x = create_program(src, "add")?;
    x.compile(None)?;
    let module = load_module_data(x.ptx()?)?;
    let function = module.function("add")?;

    //const ROWS: usize = 30;
    //const COLS: usize = 1;

    //let a = Buffer::from((&device, vec![1; ROWS*COLS]));
    //let b = Buffer::from((&device, vec![2; ROWS*COLS]));

    //let c = Buffer::<i32>::new(&device, a.len);

    let len = a.len;

    let mut min_grid_size = 0;
    let mut block_size = 0;

    unsafe {
        cuOccupancyMaxPotentialBlockSize(
            &mut min_grid_size,
            &mut block_size,
            function.0,
            0,
            0,
            len as i32,
        )
        .to_result()?;
    }

    let grid_size = (len as i32 + block_size - 1) / block_size;

    println!("min_grid: {min_grid_size}");
    println!("block size: {block_size}");
    println!("grid_size: {grid_size}");

    //let threads_per_block = 256;
    //let blocks_per_grid = (a.len + threads_per_block - 1) / threads_per_block;

    let start = Instant::now();

    unsafe {
        let params = &mut [
            &a.ptr.2 as *const u64 as *mut c_void,
            &b.ptr.2 as *const u64 as *mut c_void,
            &c.ptr.2 as *const u64 as *mut c_void,
            &len as *const usize as *mut c_void,
        ];
        cuLaunchKernel(
            function.0,
            grid_size as u32,
            1,
            1,
            block_size as u32,
            1,
            1,
            0,
            device.stream().0,
            params.as_ptr() as *mut _,
            std::ptr::null_mut(),
        )
        .to_result()?;
    };

    println!("end: {:?}", start.elapsed());

    let read = device.read(&c);
    println!("read: {read:?}");

    assert_eq!(read, vec![5, 3, 10, 10, 14]);
    Ok(())
}
