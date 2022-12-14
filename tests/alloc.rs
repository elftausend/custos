use custos::prelude::*;
use custos::{Alloc, BufFlag};

#[test]
fn test_alloc() {
    let device = CPU::new();
    let ptr = device.with_slice(&[1, 5, 4, 3, 6, 9, 0, 4]);
    let buf = Buffer {
        ptr,
        len: 8,
        device: Some(&device),
        flag: BufFlag::None,
        node: device.graph().add_leaf(8),
    };
    assert_eq!(vec![1, 5, 4, 3, 6, 9, 0, 4], device.read(&buf));
}

#[cfg(feature="wgpu")]
#[test]
fn test_wgpu_alloc() {
    let device = WGPU::new(wgpu::Backends::all()).unwrap();

    let buf = Buffer::<f32, _>::new(&device, 100);

    let buf2 = Buffer::<f32, _>::from((&device, &[1., 2., 3., 4.,]));

}