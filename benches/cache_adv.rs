use std::ops::Add;
use criterion::{criterion_group, criterion_main, Criterion};
use custos::{CPU, Buffer, set_count, cpu::CPUCache};

const SIZE: usize = 10000000;

fn add_cached<T: Default+Copy+Add<Output=T>>(device: &CPU, lhs: &[T], rhs: &[T]) {
    let len = std::cmp::min(lhs.len(), rhs.len());
    let mut out = CPUCache::get::<T>(device, len);

    for i in 0..len {
        out[i] = lhs[i] + rhs[i];
    }
}

fn add<T: Default+Copy+Add<Output=T>>(lhs: &Buffer<T>, rhs: &Buffer<T>) {
    let len = std::cmp::min(lhs.len, rhs.len);
    let mut out = vec![T::default(); SIZE];

    for i in 0..len {
        out[i] = lhs[i] + rhs[i];
    }
}

pub fn bench_buf_slice_cached(c: &mut Criterion) {
    let device = CPU::new();

    let lhs = Buffer::from((&device, vec![1.1; SIZE]));
    let rhs = Buffer::from((&device, vec![0.9; SIZE]));
    
    c.bench_function("bench buf slice cached", |bench| bench.iter(|| {
        add_cached(&device, &lhs, &rhs);
        set_count(0);
    }));
}

pub fn bench_buf_slice(c: &mut Criterion) {
    let device = CPU::new();

    let lhs = Buffer::from((&device, vec![1.1; SIZE]));
    let rhs = Buffer::from((&device, vec![0.9; SIZE]));
    
    c.bench_function("bench buf slice", |bench| bench.iter(|| {
        add(&lhs, &rhs);
    }));
}

criterion_group!(benches, bench_buf_slice_cached, bench_buf_slice);
criterion_main!(benches);