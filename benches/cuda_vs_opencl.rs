use criterion::{Criterion, criterion_main, criterion_group};
use custos::{AsDev, Matrix, set_count};

const ROWS: usize = 9000; 
const COLS: usize = ROWS;

pub fn bench_add_cl(c: &mut Criterion) {
    //let device = custos::CPU::new().select();    
    let device = custos::CLDevice::new(0).unwrap().select();

    let a = Matrix::<f32>::from((&device, (ROWS, COLS), vec![2.3; ROWS*COLS]));
    let b = Matrix::<f32>::from((&device, (COLS, ROWS), vec![1.9; ROWS*COLS]));
    
    c.bench_function("bench add cl", |bench| bench.iter(|| {
        let _d = &a + &b;
        set_count(0);
    }));
}

pub fn bench_add_cuda(c: &mut Criterion) {
    //let device = custos::CPU::new().select();    
    let device = custos::CudaDevice::new(0).unwrap().select();

    let a = Matrix::<f32>::from((&device, (ROWS, COLS), vec![2.3; ROWS*COLS]));
    let b = Matrix::<f32>::from((&device, (COLS, ROWS), vec![1.9; ROWS*COLS]));
    
    c.bench_function("bench add cuda", |bench| bench.iter(|| {
        let _d = &a + &b;
        set_count(0);
    }));
}

criterion_group!(benches, bench_add_cl, bench_add_cuda);
criterion_main!(benches);
