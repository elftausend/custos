use criterion::{Criterion, criterion_group, criterion_main};
use custos::remove_value;

pub fn remove_ptr<T: Ord>(ptrs: &mut Vec<T>, match_ptr: T) {
    for (idx, ptr) in ptrs.iter_mut().enumerate() {
        if *ptr == match_ptr {
            ptrs.swap_remove(idx);
            return;
        }
    }
}

pub fn bench_remove_value(ben: &mut Criterion) {
    ben.bench_function("bench remove ptr", |bench| bench.iter(|| {        
        let mut vec: Vec<i32> = (0..10000).into_iter().collect();
        remove_ptr(&mut vec, 5000);
    }));
}

pub fn bench_remove_value_binary(ben: &mut Criterion) {
    ben.bench_function("bench remove binary", |bench| bench.iter(|| {
        let mut vec: Vec<i32> = (0..10000).into_iter().collect();
        remove_value(&mut vec, &5000).unwrap();
    }));
}

criterion_group!(benches, bench_remove_value, bench_remove_value_binary);
criterion_main!(benches);