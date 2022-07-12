//use custos::{CPU, AsDev, Matrix, BaseOps, VecRead};

fn main() {
    /*let device = CPU::new();
    let a = Matrix::from(( &device, (2, 3), [1, 2, 3, 4, 5, 6]));
    let b = Matrix::from(( &device, (2, 3), [6, 5, 4, 3, 2, 1]));
    
    // specify device for operation
    let c = device.add(&a, &b);
    assert_eq!(device.read(&c), [7, 7, 7, 7, 7, 7]);

    // select() ... sets CPU as 'global device' 
    // -> when device is not specified in an operation, the 'global device' is used
    let device = CPU::new().select();

    let a = Matrix::from(( &device, (2, 3), [1, 2, 3, 4, 5, 6]));
    let b = Matrix::from(( &device, (2, 3), [6, 5, 4, 3, 2, 1]));

    let c = a + b;
    assert_eq!(c.read(), vec![7, 7, 7, 7, 7, 7]);*/
}