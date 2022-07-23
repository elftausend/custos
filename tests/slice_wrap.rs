use custos::{CPU, AsDev, Buffer};

#[test]
fn test_wrap_slice() {
    let _device = CPU::new().select();

    let mut slice = [1, 2, 3, 4, 5, 6];

    let mut buf = Buffer::from(&mut slice);
    buf.clear();
    assert_eq!(slice, [0; 6]);
}