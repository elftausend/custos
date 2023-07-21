use std::hint::black_box;

use criterion::{criterion_main, criterion_group, Criterion};
use custos::{set_count, get_count, Device, Buffer, module_comb::{Cached, Base, Retriever}};

const SIZE: usize = 10000;

fn bench_caching_speed(c: &mut Criterion) {
    let device = custos::CPU::new();

    let lhs: Buffer = device.buffer(vec![1.0; SIZE]);
    let rhs: Buffer = device.buffer(vec![1.0; SIZE]);

    let prev_count = get_count();
    let mut group = c.benchmark_group("caching_speed");
    group.bench_function("bench_old_caching", |bench| {
        bench.iter(|| {
            let out = device.retrieve::<f32, ()>(SIZE, (&lhs, &rhs));
            let out = device.retrieve::<f32, ()>(SIZE, (&out, &rhs));
            let out = device.retrieve::<f32, ()>(SIZE, (&out, &lhs));
            let out = device.retrieve::<f32, ()>(SIZE, (&out, &lhs));
            let out = device.retrieve::<f32, ()>(SIZE, (&out, &lhs));
            let out = device.retrieve::<f32, ()>(SIZE, (&out, &lhs));
            let out = device.retrieve::<f32, ()>(SIZE, (&out, &lhs));
            let out = device.retrieve::<f32, ()>(SIZE, (&out, &lhs));
            let out = device.retrieve::<f32, ()>(SIZE, (&out, &lhs));
            let _out = black_box(device.retrieve::<f32, ()>(SIZE, (&out, &rhs)));

            unsafe {set_count(prev_count)}
        })
    });

    let device = custos::module_comb::CPU::<Cached<Base>>::new();

    group.bench_function("bench_track_caller_caching", |bench| {
        bench.iter(|| {
            let _out = black_box(device.retrieve::<f32, ()>(SIZE));
            let _out = black_box(device.retrieve::<f32, ()>(SIZE));
            let _out = black_box(device.retrieve::<f32, ()>(SIZE));
            let _out = black_box(device.retrieve::<f32, ()>(SIZE));
            let _out = black_box(device.retrieve::<f32, ()>(SIZE));
            let _out = black_box(device.retrieve::<f32, ()>(SIZE));
            let _out = black_box(device.retrieve::<f32, ()>(SIZE));
            let _out = black_box(device.retrieve::<f32, ()>(SIZE));
            let _out = black_box(device.retrieve::<f32, ()>(SIZE));
            let _out = black_box(device.retrieve::<f32, ()>(SIZE));
        })
    });

    group.bench_function("bench_realloc", |bench| {
        bench.iter(|| {
            let _out = black_box(vec![0.0f32; SIZE]);
            let _out = black_box(vec![0.0f32; SIZE]);
            let _out = black_box(vec![0.0f32; SIZE]);
            let _out = black_box(vec![0.0f32; SIZE]);
            let _out = black_box(vec![0.0f32; SIZE]);
            let _out = black_box(vec![0.0f32; SIZE]);
            let _out = black_box(vec![0.0f32; SIZE]);
            let _out = black_box(vec![0.0f32; SIZE]);
            let _out = black_box(vec![0.0f32; SIZE]);
            let _out = black_box(vec![0.0f32; SIZE]);

        })
    });

}

criterion_group!(benches, bench_caching_speed);
criterion_main!(benches);