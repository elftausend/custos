use custos::{cached, AsDev, Buffer, ClearBuf, CPU};

#[test]
#[should_panic]
fn test_as_slice_no_device() {
    let buf = {
        let _device = CPU::new().select();
        cached::<f32>(100)
    };
    buf.as_slice();
}

#[test]
#[should_panic]
fn test_as_mut_slice_no_device() {
    let mut buf = {
        let _device = CPU::new().select();
        cached::<f32>(100)
    };
    buf.as_mut_slice();
}

#[test]
#[should_panic]
fn test_host_ptr_no_device() {
    let buf = {
        let _device = CPU::new().select();
        cached::<f32>(100)
    };
    buf.host_ptr();
}

#[cfg(feature = "opencl")]
#[test]
#[should_panic]
fn test_cl_ptr_no_device() {
    use custos::CLDevice;

    let buf = {
        let _device = CLDevice::new(0).unwrap().select();
        cached::<f32>(100)
    };
    buf.cl_ptr();
}

#[cfg(feature = "cuda")]
#[test]
#[should_panic]
fn test_cu_ptr_no_device() {
    use custos::CudaDevice;

    let buf = {
        let _device = CudaDevice::new(0).unwrap().select();
        cached::<f32>(100)
    };
    buf.cu_ptr();
}

#[test]
fn test_return_buf() {
    let mut buf = {
        let device = CPU::new().select();
        Buffer::from((&device, [1, 2, 4, 6, -4]))
    };
    assert_eq!(buf.as_slice(), vec![1, 2, 4, 6, -4]);
    let device = CPU::new();
    device.clear(&mut buf);
    assert_eq!(buf.as_slice(), &[0; 5])
}
