use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use custos::{CPU, Ident};

fn bench_cache_types(c: &mut Criterion) {
    let old_device = CPU::new();

    let mut borrowing_cache = custos::borrowing_cache::BorrowingCache::default();
    for _ in 0..120 {
        borrowing_cache.add_buf_once::<f32, _, ()>(&old_device, Ident::new_bumped(1));
    }

    let mut cache = custos::Cache::default();
    for _ in 0..120 {
        cache.add_node::<f32, ()>(&old_device, Ident::new_bumped(1), (), || ());
    }

    let mut group = c.benchmark_group("caching_speed");
    group.bench_function("bench_borrowing_cache", |bench| bench.iter(|| {
        for idx in 0..10 {
            let _buf = black_box(borrowing_cache.add_or_get::<f32, CPU, ()>(&old_device, Ident {
                idx,
                len: 1,
            }));
        }
        
    }));

    group.bench_function("bench_owning_cache", |bench| bench.iter(|| {
        for idx in 0..10 {
            let _buf = black_box(cache.get::<f32, ()>(&old_device, Ident {
                idx,
                len: 1,
            },  (), || ()));
        }
        
    }));
}

criterion_group!(benches, bench_cache_types);
criterion_main!(benches);
