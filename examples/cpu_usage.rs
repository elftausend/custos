use custos::prelude::*;

fn main() {
    let device = CPU::<Base>::new();
    let mut a = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

    // specify device for operation
    device.clear(&mut a);
    assert_eq!(a.read(), [0; 6]);

    let device = CPU::<Base>::new();

    let mut a = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

    // no need to specify the device
    a.clear();
    assert_eq!(a.read(), vec![0; 6]);
}
