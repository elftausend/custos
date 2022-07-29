use std::alloc::Layout;

use criterion::{criterion_main, criterion_group, Criterion};
use custos::{CPU, Buffer};

const SIZE: usize = 10000000;

pub fn bench_layout_alloc(c: &mut Criterion) {
    let layout = Layout::array::<f32>(SIZE).unwrap();
    
    
    c.bench_function("bench layout alloc", |bench| bench.iter(|| {
        unsafe {
            let ptr = std::alloc::alloc(layout);
            std::alloc::dealloc(ptr, layout)
        }    
        /*let raw = Box::into_raw(vec![0f32; SIZE].into_boxed_slice());
        unsafe {
            Box::from_raw(raw);
        }*/
    }));
}

pub fn bench_buf_alloc(c: &mut Criterion) {
    let device = CPU::new();

    c.bench_function("bench buf alloc", |bench| bench.iter(|| {
        let buf = Buffer::<f32>::new(&device, SIZE);
        drop(buf)
    }));
}


criterion_group!(benches, bench_layout_alloc, bench_buf_alloc);
criterion_main!(benches);