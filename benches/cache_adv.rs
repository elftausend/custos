/*
use criterion::{criterion_group, criterion_main, Criterion};
use custos::{cache::Cache, set_count, Buffer, CPU};
use std::ops::Add;

const SIZE: usize = 10000000;

fn add_cached<T: Default + Copy + Add<Output = T>>(device: &CPU, lhs: &[T], rhs: &[T]) {
    let len = std::cmp::min(lhs.len(), rhs.len());
    let mut out = Cache::get::<T, CPU>(device, len, ());

    for i in 0..len {
        out[i] = lhs[i] + rhs[i];
    }
}

fn add<T: Default + Copy + Add<Output = T>>(device: &CPU, lhs: &Buffer<T, CPU>, rhs: &Buffer<T, CPU>) {
    let len = std::cmp::min(lhs.len, rhs.len);
    let mut out = Buffer::new(device, SIZE);

    for i in 0..len {
        out[i] = lhs[i] + rhs[i];
    }
}

pub fn bench_buf_slice_cached(c: &mut Criterion) {
    let device = CPU::<Base>::new();

    let lhs = Buffer::from((&device, vec![1.1; SIZE]));
    let rhs = Buffer::from((&device, vec![0.9; SIZE]));

    c.bench_function("bench buf slice cached", |bench| {
        bench.iter(|| {
            add_cached(&device, &lhs, &rhs);
            add_cached(&device, &lhs, &rhs);
            add_cached(&device, &lhs, &rhs);
            add_cached(&device, &lhs, &rhs);
            set_count(0);
        })
    });
}

pub fn bench_buf_slice(c: &mut Criterion) {
    let device = CPU::<Base>::new();

    let lhs = Buffer::from((&device, vec![1.1; SIZE]));
    let rhs = Buffer::from((&device, vec![0.9; SIZE]));

    c.bench_function("bench buf slice", |bench| {
        bench.iter(|| {
            add(&device, &lhs, &rhs);
            add(&device, &lhs, &rhs);
            add(&device, &lhs, &rhs);
            add(&device, &lhs, &rhs);
        })
    });
}

criterion_group!(benches, bench_buf_slice_cached, bench_buf_slice);
criterion_main!(benches);
*/
fn main() {}
