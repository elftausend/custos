#[cfg(feature="opencl")]
#[test]
fn test_error() {
    use custos::{opencl::api::OCLErrorKind, CLDevice};

    let device = CLDevice::get(10000);
        
    match device {
        Ok(_) => println!("ok?"),
        Err(e) => {
            match e.kind::<OCLErrorKind>().unwrap() {
                OCLErrorKind::InvalidDeviceIdx => println!("correct"),
                _ => panic!("wrong error kind"),
            }
        },
    }
}

#[cfg(feature="opencl")]
#[test]
fn test_questionmark() -> Result<(), Box<dyn std::error::Error>> {
    use custos::CLDevice;

    let _device = CLDevice::get(0)?;  
    Ok(())
}