use custos::{libs::cpu::CPU, AsDev, Matrix};

#[test]
fn test_matrix_read() {
    let device = CPU::new().select();

    let matrix = Matrix::from(( &device, (2, 3), [1.51, 6.123, 7., 5.21, 8.62, 4.765]));
    let read = matrix.read();
    assert_eq!(&read, &[1.51, 6.123, 7., 5.21, 8.62, 4.765]);
}
