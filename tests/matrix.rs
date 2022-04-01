use custos::{libs::cpu::{CPU, each_op}, AsDev, Matrix, CLDevice};

#[test]
fn test_matrix_read() {
    let device = CPU::new().select();

    let matrix = Matrix::from(( &device, (2, 3), [1.51, 6.123, 7., 5.21, 8.62, 4.765]));
    let read = matrix.read();
    assert_eq!(&read, &[1.51, 6.123, 7., 5.21, 8.62, 4.765]);
}

#[test]
fn test_each_op() {
    let device = CPU::new().select();

    let x = Matrix::from((&device, (2, 3), [1, 2, 3, 4, 5, 6]));
    let res = each_op(&device, x, |x| x+1);
    assert_eq!(res.read(), vec![2, 3, 4, 5, 6, 7])
}

#[test]
fn test_print() {
    let device = CPU::new().select();

    let x = Matrix::from((&device, (2, 3), [1, 2, 3, 4, 5, 6]));
    println!("x: {:?}", x);
}

#[test]
fn test_sub_assign() {
    let device = CPU::new().select();

    let mut x = Matrix::from((&device, (2, 3), [1, 2, 3, 4, 5, 6]));
    let y = Matrix::from((&device, (2, 3), [3, 4, 5, 6, 7, 8]));
    
    x -= y;
    assert_eq!(x.read(), vec![-2, -2, -2, -2, -2, -2]);


    let device = CLDevice::get(0).unwrap().select();

    let mut x = Matrix::from((&device, (2, 3), [1, 2, 3, 4, 5, 6]));
    let y = Matrix::from((&device, (2, 3), [3, 4, 5, 6, 7, 8]));
    
    x -= y;
    assert_eq!(x.read(), vec![-2, -2, -2, -2, -2, -2])
}