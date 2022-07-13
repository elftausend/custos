// TODO: move to custos_math
/*
use std::ptr::null_mut;

use criterion::{Criterion, criterion_main, criterion_group};
use custos::{cpu::element_wise_op_mut, Buffer};
use custos::{CPU, AsDev, Matrix, set_count};

fn slice_add<T: Copy + std::ops::Add<Output = T>>(a: &[T], b: &[T], c: &mut [T]) {
    element_wise_op_mut(a, b, c, |a, b| a+b)
}

const SIZE: usize = 100;

pub fn bench_fixed(ben: &mut Criterion) {
    let mut slice_a = [3.123; SIZE];
    let a = Buffer {
        ptr: (slice_a.as_mut_ptr(), null_mut(), 0),
        len: slice_a.len()
    };

    let mut slice_b = [4.523; SIZE];
    let b = Buffer {
        ptr: (slice_b.as_mut_ptr(), null_mut(), 0),
        len: slice_b.len()
    };

    let mut slice_c = [0.; SIZE];
    let mut c = Buffer {
        ptr: (slice_c.as_mut_ptr(), null_mut(), 0),
        len: slice_c.len()
    };

    ben.bench_function("bench fixed", |bench| bench.iter(|| {
        slice_add(&a, &b, &mut c);
    }));
}

pub fn bench_vec(ben: &mut Criterion) {
    let device = CPU::new().select();
    let a = Matrix::from((&device, 1, SIZE, [3.123; SIZE]));
    let b = Matrix::from((&device, 1, SIZE, [4.523; SIZE]));
    ben.bench_function("bench vec", |bench| bench.iter(|| {
        let _c = &a + &b;
        set_count(0)
    }));
}

criterion_group!(benches, bench_fixed, bench_vec);
criterion_main!(benches);
*/
fn main() {}
