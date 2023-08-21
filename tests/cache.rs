#[cfg(not(feature = "realloc"))]
use std::ptr::null_mut;

#[cfg(feature = "cpu")]
#[cfg(not(feature = "realloc"))]
use custos::{Buffer, CPU};

#[cfg(feature = "cpu")]
#[cfg(not(feature = "realloc"))]
#[track_caller]
fn cached_add<'a, Mods: custos::Retrieve<CPU<Mods>, f32>>(
    device: &'a CPU<Mods>,
    a: &[f32],
    b: &[f32],
) -> Buffer<'a, f32, CPU<Mods>> {
    use custos::Retriever;

    let mut out = device.retrieve(10, ());

    for i in 0..out.len() {
        out[i] = a[i] + b[i];
    }
    out
}

#[cfg(feature = "cpu")]
#[cfg(not(feature = "realloc"))]
#[test]
fn test_caching_cpu() {
    use custos::{Base, Cached};

    let device = CPU::<Cached<Base>>::new();

    let a = Buffer::<f32, _>::new(&device, 100);
    let b = Buffer::<f32, _>::new(&device, 100);

    let mut old_ptr = null_mut();

    for _ in 0..100 {
        let mut out = cached_add(&device, &a, &b);
        if out.host_ptr() != old_ptr && !old_ptr.is_null() {
            panic!("Should be the same pointer!");
        }
        old_ptr = out.host_ptr_mut();
    }
}
