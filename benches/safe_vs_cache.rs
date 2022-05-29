/* 
use criterion::{Criterion, criterion_main, criterion_group};
use custos::{AsDev, Matrix, set_count};

pub fn bench_gemm(c: &mut Criterion) {

    //const ROWS: usize = 419;
    //const COLS: usize = 810;

    const ROWS: usize = 2usize.pow(12);
    const COLS: usize = 2usize.pow(12);

    let device = custos::CPU::new().select();
    
    //let device = custos::CLDevice::get(0).unwrap().select();
    let a = Matrix::<f32>::from((&device, (ROWS, COLS), vec![2.3; ROWS*COLS]));
    let b = Matrix::<f32>::from((&device, (COLS, ROWS), vec![4.3; COLS*ROWS]));

    //let cm = Matrix::<f32>::from((&device, (ROWS, ROWS), vec![4.9; ROWS*ROWS]));
    
    c.bench_function("bench gemm", |bench| bench.iter(|| {
        let _d = a.gemm(&b) /*+ &cm*/;
        set_count(0);
    }));
}

criterion_group!(benches, bench_gemm);
criterion_main!(benches);
*/
fn main() {}