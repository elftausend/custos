use criterion::{Criterion, criterion_main, criterion_group};
#[cfg(not(feature="safe"))]
use custos::{cpu::element_wise_op_mut, Buffer};
use custos::{CPU, AsDev, Matrix, set_count};

#[cfg(not(feature="safe"))]
fn slice_add<T: Copy + std::ops::Add<Output = T>>(a: &[T], b: &[T], c: &mut [T]) {
    element_wise_op_mut(a, b, c, |a, b| a+b)
}

#[cfg(not(feature="safe"))]
pub fn bench_fixed(ben: &mut Criterion) {
    let a: Buffer<f32> = (&mut [3.123; 1000]).into();
    let b: Buffer<f32> = (&mut [4.523; 1000]).into();
    let mut c: Buffer<f32> = (&mut [0.; 1000]).into();

    ben.bench_function("bench fixed", |bench| bench.iter(|| {
        slice_add(&a, &b, &mut c);
    }));
}

pub fn bench_vec(ben: &mut Criterion) {
    let device = CPU::new().select();
    let a = Matrix::from((&device, 1, 1000, [3.123; 1000]));
    let b = Matrix::from((&device, 1, 1000, [4.523; 1000]));

    ben.bench_function("bench vec", |bench| bench.iter(|| {
        let _ = &a + &b;
        set_count(0)
    }));
}

#[cfg(not(feature="safe"))]
criterion_group!(benches, bench_vec, bench_fixed);
criterion_group!(benches, bench_vec);
criterion_main!(benches);