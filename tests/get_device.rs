use custos::{Matrix, get_device, VecRead, libs::cpu::CPU, AsDev, GLOBAL_DEVICE};

#[test]
fn test_matrix_read() {
    CPU.sync().select();

    let read = get_device!(VecRead, f32);

    let matrix = Matrix::from(((2, 3), &[1.51, 6.123, 7., 5.21, 8.62, 4.765]));
    let read = read.read(matrix.data());
    assert_eq!(&read, &[1.51, 6.123, 7., 5.21, 8.62, 4.765]);
}