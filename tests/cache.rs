use std::ptr::null_mut;

#[cfg(feature = "cpu")]
use custos::{Buffer, CPU};

#[cfg(feature = "cpu")]
fn cached_add<'a, Mods>(device: &'a CPU<Mods>, a: &[f32], b: &[f32]) -> Buffer<'a, f32, CPU<Mods>>
where
    Mods: custos::Retrieve<'a, CPU<Mods>, f32>,
    Mods::Wrap<'a, f32, custos::cpu::CPUPtr<f32>>: custos::HostPtr<f32>,
{
    use custos::{Device, HostPtr, Retriever};

    let mut out = device.retrieve(10, ()).unwrap();

    let out_slice = out.as_mut_slice();
    for i in 0..out_slice.len() {
        out_slice[i] = a[i] + b[i];
    }
    out
}

#[cfg(feature = "cpu")]
#[cfg(feature = "cached")]
#[cfg_attr(miri, ignore)]
#[test]
fn test_caching_cpu() {
    use custos::{Base, Cached, Cursor};

    let device = CPU::<Cached<Base>>::new();

    let a = Buffer::<f32, _>::new(&device, 100);
    let b = Buffer::<f32, _>::new(&device, 100);

    let mut old_ptr = null_mut();

    for _ in device.range(0..100) {
        let out = cached_add(&device, &a, &b);
        if out.data().ptr != old_ptr && !old_ptr.is_null() {
            panic!("Should be the same pointer!");
        }
        old_ptr = out.data().ptr;
    }
}
