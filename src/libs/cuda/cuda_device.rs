use super::api::{device, create_context, CudaIntDevice, Context, cublas::{create_handle, CublasHandle}};

pub struct CudaDevice {
    device: CudaIntDevice,
    ctx: Context,
    handle: CublasHandle

}

impl CudaDevice {
    pub fn new(idx: usize) -> crate::Result<CudaDevice> {
        let device = device(idx as i32)?;
        let ctx = create_context(&device)?;
        
        let handle = create_handle()?;

        Ok(CudaDevice {
            device,
            ctx,
            handle,
        })
    }
}