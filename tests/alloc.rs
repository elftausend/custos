#[cfg(feature = "wgpu")]
use custos::Alloc;
#[cfg(feature = "wgpu")]
use custos::prelude::*;

#[cfg(feature = "wgpu")]
#[test]
fn test_wgpu_alloc() {
    let device = WGPU::new(wgpu::Backends::all()).unwrap();

    let buf = Buffer::<f32, _>::new(&device, 100);

    assert_eq!(buf.read(), &[0.; 100]);

    let buf1 = Buffer::<f32, _>::from((&device, &[1., 2., 3., 4., -9.]));

    assert_eq!(buf1.read(), &[1., 2., 3., 4., -9.])
}
