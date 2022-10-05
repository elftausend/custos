use custos::prelude::*;
use custos::{cpu::CPUPtr, Alloc, BufFlag, PtrType};

#[test]
fn test_alloc() {
    let device = CPU::new();
    let ptrs: (*mut u8, *mut std::ffi::c_void, u64) = device.with_data(&[1, 5, 4, 3, 6, 9, 0, 4]);
    let buf = Buffer {
        ptr: CPUPtr::from_ptrs(ptrs),
        len: 8,
        device: Some(&device),
        flag: BufFlag::None,
        node: device.graph().add_leaf(8),
    };
    assert_eq!(vec![1, 5, 4, 3, 6, 9, 0, 4], device.read(&buf));
}
