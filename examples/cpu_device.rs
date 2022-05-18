use custos::{CPU, BaseOps, VecRead, Matrix};

fn main() {
    let device = CPU::new();
    let a = Matrix::<f32>::new(&device, (5, 5));
    let b = Matrix::from((&device, (5, 5), vec![1.3; 5*5]));

    let out = device.add(&a, &b);

    assert_eq!(device.read(out.as_buf()), vec![1.3; 5*5]);
}