use custos::{libs::{cpu::CPU, opencl::{CLDevice, CACHE_COUNT}}, Buffer, AsDev, Matrix, Device, VecRead, range, Gemm, GLOBAL_DEVICE};


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
fn test_element_wise_add_cl() {
    let device = CLDevice::get(0).unwrap().select();

    let a = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));
    let b = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));
    
    for _ in 0..1000 {
        let c = a + b;
        assert_eq!(vec![2, 8, 4, 18], device.read(&c.data()));
        unsafe {CACHE_COUNT = 0};
    }
}

#[test]
fn test_element_wise_add_cpu() {
    CPU.sync().select();

    let a = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));
    let b = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));

    for _ in range(0..1000) {
        let c = a + b;
        assert_eq!(vec![2, 8, 4, 18], CPU.read(&c.data()));   
    }
}

#[test]
fn test_ew_add_cpu_a_cl() {
    CPU.sync().select();

    let a = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));
    let b = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));

    for _ in range(0..1000) {
        let c = a + b;
        assert_eq!(vec![2, 8, 4, 18], CPU.read(&c.data()));   
    }

    let device = CLDevice::get(0).unwrap().select();

    let a = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));
    let b = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));

    for _ in range(0..1000) {
        let c = a + b;
        assert_eq!(vec![2, 8, 4, 18], device.read(&c.data()));
        
    }
}

#[test]
fn test_gemm() {
    CPU.sync().select();

    let a = Matrix::from(( (1, 4), &[1., 4., 2., 9.] ));
    let b = Matrix::from(( (4, 1), &[5., 4., 2., 9.] ));

    let device = CLDevice::get(0).unwrap();
    
    let a_cl = Matrix::from(( device, (1, 4), &[1f32, 4., 2., 9.] ));
    let b_cl = Matrix::from(( device, (4, 1), &[5., 4., 2., 9.] ));
    
    
    
    for _ in range(0..100000) {
        
        let c1 = CPU.gemm(a, b);
        let c3 = device.gemm(a_cl, b_cl);
        let c2 = a.gemm(b);

        assert_eq!(CPU.read(&c1.data()), CPU.read(&c2.data()));
        assert_eq!(CPU.read(&c1.data()), device.read(&c3.data()));
        
    }

}
