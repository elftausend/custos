use std::ops::Range;

use custos::{libs::opencl::{CLCACHE_COUNT, CLDevice}, Matrix, AsDev, VecRead};

#[test]
fn counting() {

    for epoch in range(0..10) {
        println!("count: {}", epoch);
    }
}

#[test]
fn this() {
    let device = CLDevice::get(0).unwrap().select();

    let a = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));
    let b = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));

    let z = Matrix::from(( (1, 4), &[1, 2, 3, 4] ));

    for _ in range(0..100000) {
        let c = a + b;
        assert_eq!(vec![2, 8, 4, 18], device.read(&c.data()));
        let d = c + z;
        assert_eq!(vec![3, 10, 7, 22], device.read(&d.data()));
        //unsafe {CLCACHE_COUNT = 0};
    }
}