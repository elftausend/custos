use custos::{libs::{opencl::CLDevice, cpu::CPU}, AsDev, Matrix, range, VecRead, BaseOps};


#[test]
fn test_threading() {
    CLDevice::get(0).unwrap().select();

    let h1 = std::thread::spawn(|| {
        let a = Matrix::from( ( CPU, (3, 2), &[3f32, 2., 1., 5., 6., 4.]) );
        let b = Matrix::from( ( CPU, (2, 3), &[1., 3., 2., 6., 5., 4.]) );
        
        for _ in range(5000) {
            let c = CPU.mul(a, b);
           // println!("par {:?}", GLOBAL_DEVICE.lock().unwrap().cl_device);
            assert_eq!(CPU.read(c.data()), vec![3., 6., 2., 30., 30., 16.]);
        }

    });

    let h2 = std::thread::spawn(|| {
        
        let a = Matrix::from( ((3, 2), &[3f32, 2., 1., 5., 6., 4.]) );
        let b = Matrix::from( ((2, 3), &[1., 3., 2., 6., 5., 4.]) );
        
        for _ in range(5000) {
            let c = a + b;
            
            //println!("par 1 {:?}", GLOBAL_DEVICE.lock().unwrap().cl_device);
            assert_eq!(c.read(), vec![4., 5., 3., 11., 11., 8., ]);
        }
        
    });
    h2.join().unwrap();
    h1.join().unwrap();
    
}