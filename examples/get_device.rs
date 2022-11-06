use custos::{Buffer, VecRead, CPU};

fn main() -> custos::Result<()> {
    let device = CPU::new();

    let buf = Buffer::from((&device, [1.51, 6.123, 7., 5.21, 8.62, 4.765]));

    let read = device.read(&buf);

    assert_eq!(&read, &[1.51, 6.123, 7., 5.21, 8.62, 4.765]);
    Ok(())
}
