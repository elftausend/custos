use custos::{libs::cpu::CPU, AsDev, range, Buffer};
#[cfg(feature="opencl")]
use custos::opencl::cl_device::CLDevice;

#[test]
fn test_rc_get_dev() {
    {
        let device = CPU::new().select();
        let mut a = Buffer::from(( &device, [1., 2., 3., 4., 5., 6.,]));

        for _ in range(100) {
            a.clear();
            assert_eq!(&[0.; 6], a.as_slice());
        }
        
    }    
}

#[cfg(feature="opencl")]
#[test]
fn test_dealloc_cl() {
    let device = CLDevice::new(0).unwrap().select();

    let _a = Buffer::from(( &device, [1f32, 2., 3., 4., 5., 6.,]));
    let _b = Buffer::from(( &device, [6., 5., 4., 3., 2., 1.,]));

    drop(device);

}