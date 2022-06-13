
use criterion::{Criterion, criterion_main, criterion_group};
use custos::{AsDev, Matrix, set_count};

pub fn bench_gemm(c: &mut Criterion) {
    const ROWS: usize = 9000; 
    const COLS: usize = ROWS;

    //let device = custos::CPU::new().select();    
    let device = custos::CLDevice::new(0).unwrap().select();

    let a = Matrix::<f32>::from((&device, (ROWS, COLS), vec![2.3; ROWS*COLS]));
    let b = Matrix::<f32>::from((&device, (COLS, ROWS), vec![1.9; ROWS*COLS]));
    
    c.bench_function("bench gemm", |bench| bench.iter(|| {
        let _d = a.gemm(&b);
        set_count(0);
    }));
}

criterion_group!(benches, bench_gemm);
criterion_main!(benches);
