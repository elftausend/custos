use custos::{Error, DeviceError};

#[cfg(feature="opencl")]
#[test]
fn test_error() {
    use custos::{opencl::api::OCLErrorKind, CLDevice};

    let device = CLDevice::new(1000000000000000000);
        
    match device {
        Ok(_) => println!("ok?"),
        Err(e) => {
            if e.kind() == Some(&OCLErrorKind::InvalidDeviceIdx) {
                println!("correct");
            } else {
                panic!("wrong error kind")
            }
            match e.kind::<OCLErrorKind>().unwrap() {
                OCLErrorKind::InvalidDeviceIdx => println!("correct"),
                _ => panic!("wrong error kind"),
            }
        },
    }
}

#[cfg(feature="opencl")]
#[test]
#[should_panic]
fn test_error_panics() {
    use custos::CLDevice;
    CLDevice::new(10000000000000000).unwrap();
    
}

#[cfg(feature="opencl")]
#[test]
fn test_questionmark() -> Result<(), Box<dyn std::error::Error>> {
    use custos::CLDevice;

    let _device = CLDevice::new(0)?;  
    Ok(())
}

#[test]
fn test_print_error() {
    let err = Error::from(DeviceError::NoDeviceSelected);
    assert_eq!("No device selected, .select() on a device was not called before get_device! call", &format!("{err}"));
    assert_eq!("No device selected, .select() on a device was not called before get_device! call", &format!("{err:?}"));
}

#[test]
fn test_std_err() {
    let err = Error::from(DeviceError::NoDeviceSelected);
    let e: Box<dyn std::error::Error> = err.into();
    assert_eq!(e.downcast_ref::<DeviceError>(), Some(&DeviceError::NoDeviceSelected));
}

#[cfg(feature="opencl")]
#[test]
fn test_ocl_errors() {
    use custos::opencl::api::OCLErrorKind;

    for i in -70..=-1 {
        let err = OCLErrorKind::from_value(i);
        println!("err: {err}");
    }
}