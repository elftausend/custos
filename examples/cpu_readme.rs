use custos::{AsDev, Buffer, ClearBuf, VecRead, CPU};

fn main() {
    let device = CPU::new();
    let mut a = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

    // specify device for operation
    device.clear(&mut a);
    assert_eq!(device.read(&a), [0; 6]);

    // select() ... sets CPU as 'global device'
    // -> when device is not specified in an operation, the 'global device' is used
    let device = CPU::new().select();

    let mut a = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

    // no need to specify the device
    a.clear();
    assert_eq!(a.read(), vec![0; 6]);
}
