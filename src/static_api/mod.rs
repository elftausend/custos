use crate::CPU;

mod iter;
mod to_device;

pub trait GPU {}

#[cfg(feature="opencl")]
impl GPU for crate::OpenCL {}
#[cfg(feature="cuda")]
impl GPU for crate::CUDA {}

thread_local! {
    pub static GLOBAL_CPU: CPU = {
        
        CPU::new()
    };
}

#[inline]
pub fn static_cpu() -> &'static CPU {
    // Safety: GLOBAL_CPU should live long enough
    unsafe {
        GLOBAL_CPU
            .with(|device| device as *const CPU)
            .as_ref()
            .unwrap()
    }
}

#[cfg(feature="opencl")]
thread_local! {
    pub static GLOBAL_OPENCL: crate::OpenCL = {
        let idx = std::env::var("CUSTOS_CL_DEVICE_IDX")
            .unwrap_or("0".into())
            .parse()
            .expect("Environment variable 'CUSTOS_CL_DEVICE_IDX' contains an invalid opencl device index!");

        crate::OpenCL::new(idx).expect("Could not create a static OpenCL device.")
    };
}

#[cfg(feature="opencl")]
#[inline]
pub fn static_opencl() -> &'static crate::OpenCL {
    // Safety: GLOBAL_OPENCL should live long enough
    unsafe {
        GLOBAL_OPENCL
            .with(|device| device as *const crate::OpenCL)
            .as_ref()
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::Buffer;

    #[test]
    fn test_to_cl() {
        let buf = Buffer::from(&[1f32, 2., 3.,]);
        let mut cl = buf.to_cl();
        
        assert_eq!(cl.read(), vec![1., 2., 3.,]);

        cl.clear();

        assert_eq!(cl.read(), vec![0.; 3]);
    }

    #[test]
    fn test_to_cpu() {
        let buf = Buffer::from(&[2f32, 5., 1.,]).to_cl();

        let buf = buf.to_cpu();
    }

    #[test]
    fn test_to_gpu() {
        let buf = Buffer::from(&[2f32, 5., 1.,]).to_gpu();
    }
}