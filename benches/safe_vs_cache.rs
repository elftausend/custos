use criterion::{Criterion, criterion_main, criterion_group};
use custos::{CPU, AsDev, Matrix, set_count};

pub fn bench_gemm(c: &mut Criterion) {
    let device = CPU::new().select();
    //let device = custos::CLDevice::get(0).unwrap().select();
    let a = Matrix::<f32>::from((&device, (50000, 1000), vec![2.3; 50000*1000]));
    let b = Matrix::<f32>::from((&device, (1000, 1000), vec![4.3; 1000*1000]));

    let cm = Matrix::<f32>::from((&device, (50000, 1000), vec![4.9; 50000*1000]));
    
    c.bench_function("bench gemm", |bench| bench.iter(|| {
        let _d = a.gemm(&b) + &cm;
        set_count(0);
    }));
}

criterion_group!(benches, bench_gemm);
criterion_main!(benches);