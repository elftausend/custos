use custos::{AsDev, libs::{cpu::CPU, opencl::{CACHE_COUNT, CLDevice}}, Matrix, range, VecRead};

#[test]
fn test_range() {
    let mut count = 0;
    for epoch in range(10) {
        assert_eq!(epoch, count);
        count += 1;
    }
}

#[test]
fn test_use_range_for_ew_add() {
    let device = CLDevice::get(0).unwrap().select();

    let a = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));
    let b = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));

    let z = Matrix::from(( (1, 4), &[1, 2, 3, 4] ));

    for _ in range(100) {
        let c = a + b;
        assert_eq!(vec![2, 8, 4, 18], device.read(c.data()));
        let d = c + z;
        assert_eq!(vec![3, 10, 7, 22], device.read(d.data()));
        
        assert!(unsafe {CACHE_COUNT == 2});
    }

    assert!(unsafe {CACHE_COUNT == 0});

    let a = Matrix::from(( (1, 5), &[1, 4, 2, 9, 1] ));
    let b = Matrix::from(( (1, 5), &[1, 4, 2, 9, 1] ));

    let z = Matrix::from(( (1, 5), &[1, 2, 3, 4, 5] ));

    for _ in range(100) {
        let c = a + b;
        assert_eq!(vec![2, 8, 4, 18, 2], device.read(c.data()));
        let d = c + z;
        assert_eq!(vec![3, 10, 7, 22, 7], device.read(d.data()));

        assert!(unsafe {CACHE_COUNT == 2});

    }
    assert!(unsafe {CACHE_COUNT == 0});
}

#[test]
fn test_nested_for() {
    CPU.select();
    
    let a = Matrix::from(( (1, 5), &[1, 4, 2, 9, 1] ));
    let b = Matrix::from(( (1, 5), &[1, 4, 2, 9, 1] ));   

    for _ in range(100) {
        let c = a + b;
        for _ in range(200) {
            let d = c + b;
            let e =  a + b + c + d;
            assert!(unsafe {CACHE_COUNT == 5});

            for _ in range(10) {
                let _ = d + e;
                assert!(unsafe {CACHE_COUNT == 6});
            }

        }
        assert!(unsafe {CACHE_COUNT == 1})
    }

    assert!(unsafe {CACHE_COUNT == 0});
}