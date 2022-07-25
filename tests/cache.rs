use std::ptr::null_mut;

use custos::{cached, CacheBuffer, cpu::CPU_CACHE, range, AsDev, Buffer, CPU};

fn cached_add(a: &[f32], b: &[f32]) -> Buffer<f32> {
    let mut out = cached(a.len());
    for i in 0..out.len {
        out[i] = a[i] + b[i];
    }
    out
}

#[test]
fn test_caching_cpu() {
    let device = CPU::new().select();

    let a = Buffer::<f32>::new(&device, 100);
    let b = Buffer::<f32>::new(&device, 100);

    let mut old_ptr = null_mut();

    for _ in range(100) {
        let out = cached_add(&a, &b);
        if out.host_ptr() != old_ptr && !old_ptr.is_null() {
            panic!("Should be the same pointer!");
        }
        old_ptr = out.host_ptr();
        let len = CPU_CACHE.with(|cache| cache.borrow().nodes.len());
        assert_eq!(len, 1);
    }
}
