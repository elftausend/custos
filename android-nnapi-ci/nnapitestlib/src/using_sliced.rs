use custos::{NnapiDevice, WithShape};
use ndk_sys::android_LogPriority;
use sliced::Matrix;

use crate::log;


pub fn run_sliced_model() -> custos::Result<String> {
    let device = NnapiDevice::new()?;

    let a = Matrix::with(&device, [[1, 3, 4], [1, 3, 4,]]);
    let b = Matrix::with(&device, [[1, 3, 4], [1, 3, 4,]]);

    let x = a + b;
    // let x = device.add(&a, &b);
    let result = device.run(x.to_buf());

    log(android_LogPriority::ANDROID_LOG_DEBUG, &format!("{result:?}"));
    
    Ok("sliced".into())

}
