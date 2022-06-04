#[cfg(not(feature="safe"))]
use std::time::Instant;

#[cfg(not(feature="safe"))]
use custos::{Buffer, Matrix, CPU, AsDev, cpu::element_wise_op_mut, range};

#[cfg(not(feature="safe"))]
fn slice_add<T: Copy + std::ops::Add<Output = T>>(a: &[T], b: &[T], c: &mut [T]) {
    element_wise_op_mut(a, b, c, |a, b| a+b)
}
#[cfg(not(feature="safe"))]
fn main() {
    let device = CPU::new().select();

    const TIMES: usize = 10000;

    let start = Instant::now();


    let a: Buffer<f32> = (&mut [3.123; 1000]).into();
    let b: Buffer<f32> = (&mut [4.523; 1000]).into();
    let mut c: Buffer<f32> = (&mut [0.; 1000]).into();

    
    for _ in range(TIMES) {
        slice_add(&a, &b, &mut c);
    }
        
    println!("duration: {:?}", start.elapsed() / TIMES as u32);

    let start = Instant::now();

    let a = Matrix::from((&device, 1, 1000, [3.123; 1000]));
    let b = Matrix::from((&device, 1, 1000, [4.523; 1000]));

    for _ in range(TIMES) {
        let _ = &a + &b;
    }
    
    
    
    println!("duration: {:?}", start.elapsed());
    
}

#[cfg(feature="safe")]
fn main() {}