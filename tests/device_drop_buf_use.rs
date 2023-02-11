//use custos::{Buffer, ClearBuf, CPU, WithConst};

/*
#[test]
fn test_device_return() {

    let x = {
        let device = CPU::new();
        let buf = Buffer::with(&device, [1., 2., 3.,]);
        buf.device()
    };
}
*/

/*
// won't compile
#[test]
fn test_return_buf() {
    let mut buf = {
        let device = CPU::new();
        Buffer::from((&device, [1, 2, 4, 6, -4]))
    };
    assert_eq!(buf.as_slice(), vec![1, 2, 4, 6, -4]);
    let device = CPU::new();
    device.clear(&mut buf);
    assert_eq!(buf.as_slice(), &[0; 5])
}*/

/*
// won't compile
#[test]
fn test_return_cache_buf() {
    let mut buf = {
        let device = CPU::new();
        custos::cpu::cpu_cached::<i32>(&device, 100)
    };
    assert_eq!(buf.as_slice(), vec![1, 2, 4, 6, -4]);
    let device = CPU::new();
    device.clear(&mut buf);
    assert_eq!(buf.as_slice(), &[0; 5])
}
*/

/*
// won't compile
#[test]
fn test_return_cache_dev() {
    let mut buf = {
        let device = CPU::new();
        let dev = custos::AsDev::dev(&device);
        custos::cached::<i32>(&dev, 10)
    };
    assert_eq!(buf.as_slice(), vec![1, 2, 4, 6, -4]);
    let device = CPU::new();
    device.clear(&mut buf);
    assert_eq!(buf.as_slice(), &[0; 5])
}
*/

/*
// won't compile
#[test]
fn test_return_cache_dev() {
    let mut buf = {
        let device = CPU::new();
        let a = Buffer::<f32>::new(&device, 10);
        let dev = a.device;
        custos::cached::<i32>(&dev, 10)
    };
    assert_eq!(buf.as_slice(), vec![1, 2, 4, 6, -4]);
    let device = CPU::new();
    device.clear(&mut buf);
    assert_eq!(buf.as_slice(), &[0; 5])
}
*/

/*
// won't compile
#[test]
fn test_clone_buf_invalid_return() {
    {
        let device = CPU::new();
        let buf = Buffer::<f32>::new(&device, 10);
        buf.clone()
    };
}*/

// should not compile, but it does (unsafe block)
/*
use custos::{CPU, Buffer};

#[test]
fn test_shallow_ub() {
    let device = CPU::new();

    let _x = {
        let buf: Buffer = Buffer::from((&device, vec![1f32, 2., 3., 4., 5.]));
        let x: Buffer = unsafe {buf.shallow()};
        x
    };

    //drop(device);
    //println!("x: {x:?}");
}
*/

/*
use custos::{CPU, Buffer};

#[test]
fn test_as_dims_transform() {
    let device = CPU::new();

    let _x = {
        let buf: Buffer = Buffer::from((&device, vec![1f32, 2., 3., 4., 5.]));
        buf.as_dims::<()>()
    };

    //drop(device);
    //println!("x: {x:?}");
}*/
