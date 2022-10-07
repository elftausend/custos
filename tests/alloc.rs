use custos::prelude::*;
use custos::{Alloc, BufFlag};

#[test]
fn test_alloc() {
    let device = CPU::new();
    let ptr = device.with_data(&[1, 5, 4, 3, 6, 9, 0, 4]);
    let buf = Buffer {
        ptr,
        len: 8,
        device: Some(&device),
        flag: BufFlag::None,
        node: device.graph().add_leaf(8),
    };
    assert_eq!(vec![1, 5, 4, 3, 6, 9, 0, 4], device.read(&buf));
}
