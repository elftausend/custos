use custos::{libs::cpu::CPU, AsDev, Matrix, VecRead};
#[cfg(feature="opencl")]
use custos::CLDevice;

#[test]
fn test_matrix() {
    let device = CPU::new();
    let matrix = Matrix::<f32>::new(&device, (10, 10));
    assert_eq!(device.read(matrix.as_buf()), vec![0.; 10*10]);
}

#[test]
fn test_matrix_read() {
    let device = CPU::new().select();

    let matrix = Matrix::from(( &device, (2, 3), [1.51, 6.123, 7., 5.21, 8.62, 4.765]));
    let read = matrix.read();
    assert_eq!(&read, &[1.51, 6.123, 7., 5.21, 8.62, 4.765]);
}

#[cfg(feature="opencl")]
#[test]
fn test_each_op() {
    use custos::libs::cpu::each_op;
    CLDevice::new(0).unwrap();
    let device = CPU::new().select();

    let x = Matrix::from((&device, (2, 3), [1, 2, 3, 4, 5, 6]));
    let res = each_op(&device, &x, |x| x+1);
    assert_eq!(res.read(), vec![2, 3, 4, 5, 6, 7])
}

#[test]
fn test_print() {
    let device = CPU::new().select();

    let x = Matrix::from((&device, (2, 3), [1, 2, 3, 4, 5, 6]));
    println!("x: {:?}", x);
}

#[test]
fn test_sub_assign_cpu() {
    let device = CPU::new().select();

    let mut x = Matrix::from((&device, (2, 3), [1, 2, 3, 4, 5, 6]));
    let y = Matrix::from((&device, (2, 3), [3, 4, 5, 6, 7, 8]));
    
    x -= &y;
    assert_eq!(x.read(), vec![-2, -2, -2, -2, -2, -2]);
}

#[cfg(feature="opencl")]
#[test]
fn test_sub_assign_cl() {
    let device = CLDevice::new(0).unwrap().select();

    let mut x = Matrix::from((&device, (2, 3), [1, 2, 3, 4, 5, 6]));
    let y = Matrix::from((&device, (2, 3), [3, 4, 5, 6, 7, 8]));
    
    x -= &y;
    assert_eq!(x.read(), vec![-2, -2, -2, -2, -2, -2])
}

#[cfg(feature="cuda")]
#[test]
fn test_sub_assign_cuda() -> custos::Result<()> {
    use custos::CudaDevice;

    let device = CudaDevice::new(0)?.select();

    let mut x = Matrix::from((&device, (2, 3), [1, 2, 3, 4, 5, 6]));
    let y = Matrix::from((&device, (2, 3), [3, 4, 5, 6, 7, 8]));
    
    x -= &y;
    assert_eq!(x.read(), vec![-2, -2, -2, -2, -2, -2]);
    Ok(())
}

#[cfg(feature="cuda")]
#[test]
fn test_add_assign_cuda() -> custos::Result<()> {
    use custos::CudaDevice;

    let device = CudaDevice::new(0)?.select();

    let mut x = Matrix::from((&device, (2, 3), [1, 2, 3, 4, 5, 6]));
    let y = Matrix::from((&device, (2, 3),     [3, 4, 5, 6, 7, 8]));
    
    x += &y;
    assert_eq!(x.read(), vec![4, 6, 8, 10, 12, 14]);
    Ok(())
}

#[test]
fn test_debug_fmt() {
    let device = CPU::new().select();

    let x = Matrix::<i32>::from((&device, (2, 3), [1, 2, 3, 4, 5, 6]));    
    println!("{x:?}");
    println!("x");
}

fn slice_fn<T>(x: &[T]) -> &T {
    &x[0]
}

#[test]
fn test_deref() {
    let device = CPU::new().select();

    let x = Matrix::<i32>::from((&device, (1, 3), [9, 3, 4]));
    let item = *slice_fn(&x);
    assert_eq!(9, item);
}

#[test]
fn test_range_gemm() {

    let m = 3;
    let k = 2;
    let n = 3;

    let device = CPU::new().select();

    let a = Matrix::from((&device, m, k, (0..m*k).map(|x| x as f32).collect::<Vec<_>>()));
    let b = Matrix::from((&device, k, m, (0..k*n).rev().map(|x| x as f32).collect::<Vec<_>>()));

    let c = a.gemm(&b);

    // [2.0, 1.0, 0.0, 16.0, 11.0, 6.0, 30.0, 21.0, 12.0]
    println!("{c:?}");
}


