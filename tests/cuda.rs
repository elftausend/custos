
#[cfg(feature="cuda")]
#[test]
fn test_alloc() {
    use custos::cuda::api::{cmalloc, cuInit, device, create_context};

    unsafe { cuInit(0) };
    let device = device(0).unwrap();
    let _ctx = create_context(device).unwrap();

    let x = unsafe {   
        cmalloc::<f32>(10)
    }.unwrap();
}