
#[cfg(feature="cuda")]
#[test]
fn test_cuda_alloc() {
    use custos::cuda::api::{cumalloc, cuInit, device, create_context};

    unsafe { cuInit(0) };
    let device = device(0).unwrap();
    let _ctx = create_context(device).unwrap();

    let x = cumalloc::<f32>(10).unwrap();

}
#[cfg(feature="cuda")]
#[test]
fn test_cuda_alloc2() -> custos::Result<()> {
    use custos::cuda::api::{cumalloc, cuInit, device, create_context, device_count};

    unsafe { cuInit(0) };
    println!("count: {}", device_count()?);

    let device = device(1)?;
    let _ctx = create_context(device)?;

    let x = cumalloc::<f32>(10)?;

    Ok(())
}

#[cfg(feature="cuda")]
#[test]
fn test_cuda_write() -> custos::Result<()> {
    use custos::cuda::api::{cumalloc, cuInit, device, create_context, cuwrite, curead};

    unsafe { cuInit(0) };

    let device = device(0).unwrap();
    let _ctx = create_context(device).unwrap();

    let x = cumalloc::<f32>(10).unwrap();
    cuwrite(x, &[4f32;10]).unwrap();
    

    let mut read = vec![0f32; 10];
    curead(&mut read, x).unwrap();
    println!("read: {read:?}");

    Ok(())
}