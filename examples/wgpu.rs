use custos::{Buffer, WGPU};

fn main() -> custos::Result<()> {
    let device = WGPU::new(wgpu::Backends::all())?;

    let mut a = Buffer::from((&device, [5, 3, 2, 4, 6, 2]));
    a.clear();

    assert_eq!(a.read(), [0; 6]);
    Ok(())
}
