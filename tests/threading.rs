use custos::{libs::{opencl::{CLDevice, api::OCLError, CL_CACHE}, cpu::{CPU, CPU_CACHE}}, AsDev, Matrix, Threaded};

/* 
#[test]
fn test_threading() {
    CLDevice::get(0).unwrap().select();

    let h1 = std::thread::spawn(|| {
        CPU.mt::<f32>();
        let a = Matrix::from( ( CPU, (3, 2), &[3f32, 2., 1., 5., 6., 4.]) );
        let b = Matrix::from( ( CPU, (2, 3), &[1., 3., 2., 6., 5., 4.]) );
        for _ in range(5000) {
            
            let c = CPU.mul(a, b);
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
*/

#[test]
fn test_threaded_drop() -> Result<(), OCLError> {
    {
        let device = CLDevice::get(0)?.select();
        let threaded = Threaded::new(device);
        
        let a = Matrix::<f32>::new(threaded.device, (100, 100));
        let b = Matrix::<f32>::new(threaded.device, (100, 100));
    
        let c = a + b;
        let _ = a * c + b;

        {
            let threaded = Threaded::new(CPU.select());
            let d = Matrix::<f32>::new(threaded.device, (50, 12));
            let e = Matrix::<f32>::new(threaded.device, (50, 12));

            let f = d + e * d;
            let _ = f - e * f;
            assert!(CPU_CACHE.lock().unwrap().nodes.len() == 4);
        }
        //assert!(CPU_CACHE.lock().unwrap().nodes.len() == 0);
        assert!(CL_CACHE.lock().unwrap().output_nodes.len() == 3);
    }
    assert!(CL_CACHE.lock().unwrap().output_nodes.len() == 0);
    
    Ok(())
}

#[test]
fn test_threaded_drop_2() {
    std::thread::spawn(|| {
        let device = CLDevice::get(0).unwrap().select();
        let threaded = Threaded::<_>::new(device);
        
        let a = Matrix::<f32>::new(threaded.device, (100, 100));
        let b = Matrix::<f32>::new(threaded.device, (100, 100));
    
        let c = a + b;
        let _ = a * c + b;
    });
    assert!(CPU_CACHE.lock().unwrap().nodes.len() == 0);
    std::thread::spawn(|| {
        let threaded = Threaded::new(CPU.select());
        let d = Matrix::<f32>::new(threaded.device, (50, 12));
        let e = Matrix::<f32>::new(threaded.device, (50, 12));

        let f = d + e * d;
        let _ = f - e * f;
        assert!(CPU_CACHE.lock().unwrap().nodes.len() == 4);
    });
    assert!(CPU_CACHE.lock().unwrap().nodes.len() == 0);
}