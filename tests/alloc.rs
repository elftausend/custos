use custos::prelude::*;
use custos::Alloc;

#[cfg(feature = "cpu")]
#[test]
fn test_alloc() {
    let device = CPU::<Base>::new();
    let data = Alloc::<i32>::alloc_from_slice::<()>(&device, &[1, 5, 4, 3, 6, 9, 0, 4]);
    //let data = device.with_slice(&[1, 5, 4, 3, 6, 9, 0, 4]);
    let buf: Buffer<i32, CPU, ()> = Buffer {
        data,
        device: Some(&device),
    };
    assert_eq!(vec![1, 5, 4, 3, 6, 9, 0, 4], device.read(&buf));
}

#[cfg(feature = "wgpu")]
#[test]
fn test_wgpu_alloc() {
    let device = WGPU::new(wgpu::Backends::all()).unwrap();

    let buf = Buffer::<f32, _>::new(&device, 100);

    assert_eq!(buf.read(), &[0.; 100]);

    let buf1 = Buffer::<f32, _>::from((&device, &[1., 2., 3., 4., -9.]));

    assert_eq!(buf1.read(), &[1., 2., 3., 4., -9.])
}
