use custos::{get_device, AsDev, Buffer, VecRead, CPU};

fn main() -> custos::Result<()> {
    let device = CPU::new();

    let read = get_device!(device.as_dev(), VecRead<f32>);

    let buf = Buffer::from((&device, [1.51, 6.123, 7., 5.21, 8.62, 4.765]));
    let read = read.read(&buf);

    assert_eq!(&read, &[1.51, 6.123, 7., 5.21, 8.62, 4.765]);
    Ok(())
}
