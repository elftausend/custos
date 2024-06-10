use custos::{DeviceError, Error};

#[cfg(feature = "opencl")]
#[test]
fn test_error() {
    use custos::{Base, ErrorKind, OpenCL};
    use min_cl::api::OCLErrorKind;

    let device = OpenCL::<Base>::new(1000000000000000000);

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
        }
    }
}

#[cfg(feature = "opencl")]
#[test]
#[should_panic]
fn test_error_panics() {
    use custos::{Base, OpenCL};
    OpenCL::<Base>::new(10000000000000000).unwrap();
}

#[cfg(feature = "opencl")]
#[test]
fn test_questionmark() -> Result<(), Box<dyn std::error::Error + Sync + Send>> {
    use custos::{prelude::chosen_cl_idx, Base, OpenCL};

    let _device = OpenCL::<Base>::new(chosen_cl_idx())?;
    Ok(())
}

#[cfg(feature = "std")]
#[test]
fn test_print_error() {
    let err = custos::Error::from(DeviceError::UnifiedConstructInvalidInputBuffer);
    assert_eq!(
        "Only a non-drop buffer can be converted to a CPU+OpenCL buffer.",
        &format!("{err}")
    );
    assert_eq!(
        "Only a non-drop buffer can be converted to a CPU+OpenCL buffer.",
        &format!("{err:?}")
    );
}

#[cfg(feature = "std")]
#[test]
fn test_std_err() {
    let err = Error::from(DeviceError::UnifiedConstructInvalidInputBuffer);
    assert_eq!(
        err.downcast_ref::<DeviceError>(),
        Some(&DeviceError::UnifiedConstructInvalidInputBuffer)
    );
}

#[cfg(feature = "opencl")]
#[test]
fn test_ocl_errors() {
    use min_cl::api::OCLErrorKind;

    for i in -70..=-1 {
        let err = OCLErrorKind::from_value(i);
        println!("err: {err}");
    }
}
