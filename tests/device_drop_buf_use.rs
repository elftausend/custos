use custos::{Buffer, ClearBuf, CPU};

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
}

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