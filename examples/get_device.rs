use custos::{Error, CPU, get_device, Matrix, VecRead, AsDev};

fn main() -> Result<(), Error> {
    let device = CPU::new().select();

    let read = get_device!(VecRead, f32)?;

    let matrix = Matrix::from(( &device, (2, 3), [1.51, 6.123, 7., 5.21, 8.62, 4.765]));
    let read = read.read(matrix.data());
    
    assert_eq!(&read, &[1.51, 6.123, 7., 5.21, 8.62, 4.765]);
    Ok(())
}