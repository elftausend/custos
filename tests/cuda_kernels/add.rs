
#[cfg(feature="cuda")]
#[test]
fn test_add_cuda() -> custos::Result<()> {
    use custos::{CudaDevice, Buffer, cuda::api::load_module};

    let device = CudaDevice::new(0)?;

    let a = Buffer::from((&device, [1, 2, 3, 4, 5,]));
    let b = Buffer::from((&device, [4, 1, 7, 6, 9,]));

    let module = load_module("tests/cuda_kernels/add.ptx")?;

    Ok(())
}