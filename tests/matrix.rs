use custos::{libs::cpu::{CPU, each_op}, AsDev, Matrix};

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
