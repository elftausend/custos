use custos::{libs::{cpu::CPU, opencl::{CLDevice, api::OCLError, CLCACHE_COUNT}}, Buffer, AsDev, Matrix, Device, VecRead};


/* 
#[test]
fn add() -> Result<(), OCLError> {
    
    let device = CPU.select();
    
    let lhs = Buffer::from((&device, [4., 1., 2.,]));
    let rhs = Buffer::from((&device, [4., 1., 2.,]));

    let native = lhs + rhs;



    let device = CLDevice::get(0)?.select();
    
    let lhs = Buffer::from((&device, [4., 1., 2.,]));
    let rhs = Buffer::from((&device, [4., 1., 2.,]));

    let opencl = lhs + rhs;
    
    assert_eq!(opencl, native);
    Ok(())   
}
*/

pub fn read<T, D: Device<T>>(device: D, buf: &Buffer<T>) -> Vec<T> where D: VecRead<T> {
    device.read(buf)
}

#[test]
fn test_element_wise_add() {
    let device = CLDevice::get(0).unwrap().select();

    let a = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));
    let b = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));

    for _ in 0..1000 {
        let c = a + b;
        assert_eq!(vec![2, 8, 4, 18], device.read(&c.data()));
        unsafe {CLCACHE_COUNT = 0};
    }
    
}
