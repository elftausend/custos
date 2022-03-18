use custos::{AsDev, Buffer, Device, Gemm, libs::{cpu::CPU, opencl::{CACHE_COUNT, CLDevice}}, Matrix, number::Float, range, VecRead};

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

pub fn read<T, D: Device<T>>(device: D, buf: Buffer<T>) -> Vec<T> where D: VecRead<T> {
    device.read(buf)
}

#[test]
fn test_element_wise_add_cl() {
    let device = CLDevice::get(0).unwrap().select();

    let a = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));
    let b = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));
    
    for _ in 0..500 {
        let c = a + b;
        assert_eq!(vec![2, 8, 4, 18], device.read(c.data()));
        unsafe {CACHE_COUNT = 0};
    }
}

#[test]
fn test_element_wise_add_cpu() {
    CPU.sync().select();

    let a = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));
    let b = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));

    for _ in range(500) {
        let c = a + b;
        assert_eq!(vec![2, 8, 4, 18], CPU.read(c.data()));   
    }
}

#[test]
fn test_ew_add_cpu_a_cl() {
    CPU.sync().select();

    let a = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));
    let b = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));

    let c = a + b;
    assert_eq!(vec![2, 8, 4, 18], CPU.read(c.data()));   

    CLDevice::get(0).unwrap().select();

    let a = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));
    let b = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));

    let c = a + b;
    assert_eq!(vec![2, 8, 4, 18], c.read());
        
}

#[test]
fn test_ew_sub_cpu_a_cl() {
    CPU.sync().select();

    let a = Matrix::from(( (1, 4), &[1u32, 4, 2, 9] ));
    let b = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));

    let c = a - b;
    assert_eq!(vec![0, 0, 0, 0], c.read());

    CLDevice::get(0).unwrap().select();

    let a = Matrix::from(( (1, 4), &[1u32, 4, 2, 9] ));
    let b = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));

    let c = a - b;
    assert_eq!(vec![0, 0, 0, 0], c.read());

}

#[test]
fn test_ew_mul_cpu_a_cl() {
    CPU.sync().select();

    let a = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));
    let b = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));

    for _ in range(0..500) {
        let c = a * b;
        assert_eq!(vec![1, 16, 4, 81], CPU.read(c.data()));   
    }

    CLDevice::get(0).unwrap().select();

    let a = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));
    let b = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));

    for _ in range((0, 500)) {
        let c = a * b;
        assert_eq!(vec![1, 16, 4, 81], c.read());
        
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
    
    for _ in range(500) {
        
        let c1 = CPU.gemm(a, b);
        let c3 = device.gemm(a_cl, b_cl);
        let c2 = a.gemm(b);

        assert_eq!(CPU.read(c1.data()), CPU.read(c2.data()));
        assert_eq!(CPU.read(c1.data()), device.read(c3.data()));
        
    }

}

fn roughly_equal<T: Float>(lhs: &[T], rhs: &[T]) {
    for (idx, value) in lhs.iter().enumerate() {
        let diff = (*value-rhs[idx]).abs();
        if diff > T::as_generic(1E-3) {
            panic!("Slices are not roughly equal! lhs: {}, rhs: {}", value, rhs[idx])
        }
    }
}

#[test]
fn test_larger_gemm() {
    //5x7 
    let arr1 = &[
        9., 1., 3., 6., 7., 3., 63f32,
        93., 51., 23., 36., 87., 3., 63.,
        9., 1., 43., 46.3, 7., 3., 63.,
        9., 15., 73., 6.3, 7., 53., 63.,
        69., 1., 3., 6., 7., 43., 63.,   
    ];

    //7x10
    let arr2 = &[
        1f32, 2., 3., 44., 55., 6., 7., 8., 95., 103.,
        14., 2., 33., 4., 75., 6., 37., 8., 9., 120.,
        31., 2., 3., 4., 5., 6.51, 7.45, 8., 9., 10.,
        313., 244., 3., 4., 5.8, 6., 27., 48., 9., 101.,
        21., 2., 3.4324, 4., 5., 6., 75., 38., 9., 109.,
        11., 2., 3., 4., 85., 96., 7., 8., 29., 130.,
        1., 2.91, 3.909, 4., 5.634, 36., 7., 8., 9., 130.
    ];

    let should = &[2237.0f32, 1693.33, 366.2938, 728.0, 
                1264.742, 2713.53, 1271.35, 1186.0, 1662.0, 11026.0, 14711.0, 9481.33, 2692.886, 
                5144.0, 10308.742, 4307.73, 10668.35, 6898.0, 11262.0, 37628.0, 16090.899, 11606.53, 607.1938, 1049.2, 1698.482, 3215.73, 
                2657.45, 3440.4, 2384.7, 15496.3, 5246.9, 2034.53, 1189.1938, 1265.2, 6916.482, 
                8055.0303, 2668.95, 2272.4, 3870.7, 19936.3, 2737.0, 1893.33, 666.29376, 3528.0, 7964.7417, 
                6913.5303, 1971.35, 1986.0, 8522.0, 22406.0];

    let device = CLDevice::get(0).unwrap().select();

    let a = Matrix::from(((5, 7), arr1));
    let b = Matrix::from(((7, 10), arr2));

    let c = a.gemm(b);

    roughly_equal(&device.read(c.data()), should);

    CPU.sync().select();

    let a = Matrix::from(((5, 7), arr1));
    let b = Matrix::from(((7, 10), arr2));

    let cpu_c = a.gemm(b);

    roughly_equal(&device.read(c.data()), &CPU.read(cpu_c.data()));
}
