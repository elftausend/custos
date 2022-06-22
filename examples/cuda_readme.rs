use custos::{CudaDevice, Matrix, AsDev};

fn main() -> custos::Result<()> {
    let device = CudaDevice::new(0)?.select();
    let a = Matrix::from((&device, 2, 3, [5, 3, 2, 4, 6, 2]));
    let b = Matrix::from((&device, 1, 6, [1, 4, 0, 2, 1, 3]));

    let c = a + b;
    assert_eq!(c.read(), [6, 7, 2, 6, 7, 5]);

    Ok(())
}