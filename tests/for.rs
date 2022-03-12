
use custos::{libs::opencl::{CLCACHE_COUNT, CLDevice}, Matrix, AsDev, VecRead, range};

#[test]
fn counting() {

    for epoch in range(10) {
        println!("count: {}", epoch);
    }
}

#[test]
fn this() {
    let device = CLDevice::get(0).unwrap().select();

    let a = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));
    let b = Matrix::from(( (1, 4), &[1, 4, 2, 9] ));

    let z = Matrix::from(( (1, 4), &[1, 2, 3, 4] ));

    for _ in range(1000) {
        let c = a + b;
        assert_eq!(vec![2, 8, 4, 18], device.read(&c.data()));
        let d = c + z;
        assert_eq!(vec![3, 10, 7, 22], device.read(&d.data()));
        //unsafe {CLCACHE_COUNT = 0};
    }

    let a = Matrix::from(( (1, 5), &[1, 4, 2, 9, 1] ));
    let b = Matrix::from(( (1, 5), &[1, 4, 2, 9, 1] ));

    let z = Matrix::from(( (1, 5), &[1, 2, 3, 4, 5] ));

    for _ in range(100000) {
        let c = a + b;
        assert_eq!(vec![2, 8, 4, 18], device.read(&c.data()));
        let d = c + z;
        assert_eq!(vec![3, 10, 7, 22], device.read(&d.data()));

    }
}