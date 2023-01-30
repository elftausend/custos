#[cfg(not(feature = "realloc"))]
use std::ptr::null_mut;

#[cfg(not(feature = "realloc"))]
use custos::{cpu::cpu_cached, range, Buffer, CPU};

#[cfg(not(feature = "realloc"))]
fn cached_add<'a>(device: &'a CPU, a: &[f32], b: &[f32]) -> Buffer<'a, f32, CPU> {
    let mut out = cpu_cached(device, a.len());
    for i in 0..out.len() {
        out[i] = a[i] + b[i];
    }
    out
}

#[cfg(not(feature = "realloc"))]
#[test]
fn test_caching_cpu() {
    let device = CPU::new();

    let a = Buffer::<f32, _>::new(&device, 100);
    let b = Buffer::<f32, _>::new(&device, 100);

    let mut old_ptr = null_mut();

    for _ in range(100) {
        let mut out = cached_add(&device, &a, &b);
        if out.host_ptr() != old_ptr && !old_ptr.is_null() {
            panic!("Should be the same pointer!");
        }
        old_ptr = out.host_ptr_mut();
        let len = device.cache.borrow().nodes.len();
        //let len = CPU_CACHE.with(|cache| cache.borrow().nodes.len());
        assert_eq!(len, 3);
    }
}
