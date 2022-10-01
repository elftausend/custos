use custos::{
    cuda::{api::culaunch_kernel, fn_cache},
    Buffer, VecRead, CUDA,
};
use std::ffi::c_void;

mod cuda;
mod cuda_kernels;
mod nvrtc;
mod occupancy;
mod scalar_ops;
//mod to_cuda;

#[test]
fn test_cached_kernel_launch() -> custos::Result<()> {
    let device = CUDA::new(0)?;

    let a = Buffer::from((&device, [1, 2, 3, 4, 5]));
    let b = Buffer::from((&device, [4, 1, 7, 6, 9]));

    let mut c = Buffer::<i32, _>::new(&device, a.len);

    let src = r#"
        extern "C" __global__ void add(int *a, int *b, int *c, int numElements)
        {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx < numElements) {
                c[idx] = a[idx] + b[idx];
            }
    }"#;

    for _ in 0..1000 {
        fn_cache(&device, src, "add")?;

        assert_eq!(device.kernel_cache.borrow().kernels.len(), 1);
    }
    let function = fn_cache(&device, src, "add")?;

    culaunch_kernel(
        &function,
        [a.len as u32, 1, 1],
        [1, 1, 1],
        &mut device.stream(),
        &mut [
            &a.ptr.2 as *const u64 as *mut c_void,
            &b.ptr.2 as *const u64 as *mut c_void,
            &mut c.ptr.2 as *mut u64 as *mut c_void,
            &a.len as *const usize as *mut c_void,
        ],
    )?;

    assert_eq!(&vec![5, 3, 10, 10, 14], &device.read(&c));
    Ok(())
}
