use custos::{libs::{opencl::CLDevice, cpu::CPU}, AsDev, Matrix, range};


#[test]
fn test_threading() {
    CLDevice::get(0).unwrap().select();

    let h1 = std::thread::spawn(|| {
        //CPU.select();
        let a = Matrix::from( ((3, 2), &[3f32, 2., 1., 5., 6., 4.]) );
        let b = Matrix::from( ((2, 3), &[1., 3., 2., 6., 5., 4.]) );
        
        for _ in range(500) {
            let c = a * b;
            assert_eq!(c.read(), vec![3., 6., 2., 30., 30., 16.]);
        }
        
    });

    let h2 = std::thread::spawn(|| {
        let a = Matrix::from( ((3, 2), &[3f32, 2., 1., 5., 6., 4.]) );
        let b = Matrix::from( ((2, 3), &[1., 3., 2., 6., 5., 4.]) );
        
        for _ in range(500) {
            let c = a + b;
            assert_eq!(c.read(), vec![4., 5., 3., 11., 11., 8., ]);
        }
        
    });

    h1.join().unwrap();
    h2.join().unwrap();
}