use std::ops::Add;

use custos::{libs::{cpu::CPU, opencl::{CLDevice, api::OCLError}}, Device, Buffer};



#[test]
fn add() -> Result<(), OCLError> {
    
    let device = CPU.select::<f32>();
    
    let lhs = Buffer::from((&device, [4., 1., 2.,]));
    let rhs = Buffer::from((&device, [4., 1., 2.,]));

    let a = lhs + rhs;

    let device = CLDevice::get(0)?.select::<f32>();
    
    let lhs = Buffer::from((&device, [4., 1., 2.,]));
    let rhs = Buffer::from((&device, [4., 1., 2.,]));

    let a = lhs + rhs;
    
    Ok(())   
}