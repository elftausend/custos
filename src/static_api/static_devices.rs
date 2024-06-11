#[cfg(feature = "std")]
use std::sync::OnceLock;

#[cfg(not(feature = "std"))]
compile_error!("The static-api feature is only available when using the std feature.");

use crate::CPU;

#[cfg(feature = "opencl")]
use crate::opencl::chosen_cl_idx;

#[cfg(feature = "cuda")]
use crate::cuda::chosen_cu_idx;

#[cfg(feature = "std")]
pub static GLOBAL_CPU: OnceLock<CPU> = OnceLock::new();

#[inline]
pub fn static_cpu() -> &'static CPU {
    GLOBAL_CPU.get_or_init(|| CPU::based())
}

// #[cfg(feature = "opencl")]
// pub static GLOBAL_OPENCL: OnceLock<std::sync::Mutex<crate::OpenCL>> = OnceLock::new();

#[cfg(feature = "opencl")]
thread_local! {
    static GLOBAL_OPENCL: crate::OpenCL = {
        crate::OpenCL::<crate::Base>::new(chosen_cl_idx()).expect("Could not create a static OpenCL device.")
    };
}

/// Returns a static `OpenCL` device.
/// You can select the index of a static [`OpenCL`](crate::OpenCL) device by setting the `CUSTOS_CL_DEVICE_IDX` environment variable.
#[cfg(feature = "opencl")]
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

#[cfg(feature = "cuda")]
thread_local! {
    static GLOBAL_CUDA: crate::CUDA = {
        crate::CUDA::<Base>::new(chosen_cu_idx()).expect("Could not create a static CUDA device.")
    };
}

/// Returns a static `CUDA` device.
/// /// You can select the index of a static [`CUDA`](crate::CUDA) device by setting the `CUSTOS_CU_DEVICE_IDX` environment variable.
#[cfg(feature = "cuda")]
#[inline]
pub fn static_cuda() -> &'static crate::CUDA {
    // Safety: GLOBAL_CUDA should live long enough
    unsafe {
        GLOBAL_CUDA
            .with(|device| device as *const crate::CUDA)
            .as_ref()
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "std")]
    use crate::Buffer;
    
    #[cfg(feature = "std")]
    #[test]
    fn test_static_cpu() {
        let buf = Buffer::from(&[1f32, 2., 3.]);
        assert_eq!(buf.read(), vec![1., 2., 3.,]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_static_cpu_thread_return() {
        use crate::{static_api::{static_cpu, GLOBAL_CPU}, Device};

        let buf = Buffer::from(&[1f32, 2., 3.]);
        GLOBAL_CPU.get().unwrap();
        assert_eq!(buf.read(), vec![1., 2., 3.,]);

        let res = std::thread::spawn(|| {
            let cpu = static_cpu();

            cpu
        }).join().unwrap();
        
        let _out = res.buffer([1, 2, 3, 4]);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_to_cl() {
        let buf = Buffer::from(&[1f32, 2., 3.]);
        let mut cl = buf.to_cl();

        assert_eq!(cl.read(), vec![1., 2., 3.,]);

        cl.clear();

        assert_eq!(cl.read(), vec![0.; 3]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_to_cuda() {
        let buf = Buffer::from(&[1f32, 2., 3.]);
        let mut cuda = buf.to_cuda();

        assert_eq!(cuda.read(), vec![1., 2., 3.,]);

        cuda.clear();

        assert_eq!(cuda.read(), vec![0.; 3]);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_to_cpu() {
        let buf = Buffer::from(&[2f32, 5., 1.]).to_cl();
        let buf = buf.to_cpu();

        assert_eq!(buf.as_slice(), &[2., 5., 1.]);
    }

    #[cfg(any(feature = "opencl", feature = "cuda"))]
    #[test]
    fn test_to_gpu() {
        use crate::buf;

        let buf = buf![2f32, 5., 1.].to_gpu();
        assert_eq!(buf.read(), vec![2., 5., 1.]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_to_device_cu() {
        use crate::CUDA;

        let buf = Buffer::from(&[3f32, 1.4, 1., 2.]).to_dev::<CUDA>();

        assert_eq!(buf.read(), vec![3., 1.4, 1., 2.]);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_to_device_cl() {
        use crate::{buf, OpenCL};

        let buf = buf![3f32, 1.4, 1., 2.].to_dev::<OpenCL>();

        assert_eq!(buf.read(), vec![3., 1.4, 1., 2.]);
    }
}
