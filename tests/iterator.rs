use custos::Buffer;

#[test]
fn test_from_iter() {
    let buf = Buffer::from_iter(0..10);
    assert_eq!(buf.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
}

#[test]
fn test_collect() {
    let buf = (0..5).into_iter().collect::<Buffer<i32, _>>();
    assert_eq!(buf.read(), vec![0, 1, 2, 3, 4]);
}
