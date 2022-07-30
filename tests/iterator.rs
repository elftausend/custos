
/*
#[test]
fn test_from_iter() {
    let _device = CPU::new().select();
    let buf = Buffer::from_iter(0..10);
    assert_eq!(buf.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
}*/

#[cfg(feature = "opencl")]
#[test]
fn test_collect_cl() -> custos::Result<()> {
    let _device = custos::CLDevice::new(0)?.select();
    let buf = (0..5).into_iter().collect::<Buffer<i32>>();
    assert_eq!(buf.read(), vec![0, 1, 2, 3, 4]);
    Ok(())
}
