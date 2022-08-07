use custos::{CPU, Buffer};

#[test]
fn test_shallow_buf_copy() {
    let device = CPU::new();

    let buf = Buffer::from((&device, [1, 2, 3, 4, 5]));
    let mut owned = unsafe {buf.shallow()};
    owned[0] = 101;

    assert_eq!(buf.as_slice(), &[101, 2, 3, 4, 5]);
}