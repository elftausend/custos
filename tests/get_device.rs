
use custos::{AsDev, BaseDevice, get_device, GLOBAL_DEVICE, libs::{cpu::{CPU, InternCPU}, opencl::{CLDevice, cl_device::InternCLDevice}}, Matrix, VecRead};

#[test]
fn test_matrix_read() {
    let device = CPU::new().select();

    let read = get_device!(VecRead, f32);

    let matrix = Matrix::from(( &device, (2, 3), [1.51, 6.123, 7., 5.21, 8.62, 4.765]));
    let read = read.read(matrix.data());
    assert_eq!(&read, &[1.51, 6.123, 7., 5.21, 8.62, 4.765]);

    let device = CLDevice::get(0).unwrap().select();

    let read = get_device!(VecRead, f32);

    let matrix = Matrix::from(( &device, (2, 3), [1.51, 6.123, 7., 5.21, 8.62, 4.765]));
    let read = read.read(matrix.data());
    assert_eq!(&read, &[1.51, 6.123, 7., 5.21, 8.62, 4.765]);
    
    let base_device = get_device!(BaseDevice, f32);
    assert_eq!(&read, &base_device.read(matrix.data()));
    
}