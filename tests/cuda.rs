
#[cfg(feature="cuda")]
#[test]
fn test_alloc() {
    use custos::cuda::api::{cmalloc, cuInit};

    unsafe { cuInit(0) };

    let x = unsafe {   
        cmalloc::<f32>(10)
    }.unwrap();
}