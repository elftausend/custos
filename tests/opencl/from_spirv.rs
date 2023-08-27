use std::mem::size_of_val;

use min_cl::{
    api::{build_program, Program},
    CLDevice,
};
use naga::back::spv::{Options, PipelineOptions};

#[test]
fn test_opencl_from_spirv() {
    let src = "@group(0)
            @binding(0)
            var<storage, read_write> a: array<f32>;
            
            @group(0)
            @binding(1)
            var<storage, read_write> b: array<f32>;
    
            @group(0)
            @binding(2)
            var<storage, read_write> out: array<f32>;
            
            
            @compute
            @workgroup_size(1)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                out[global_id.x] = a[global_id.x] + b[global_id.x];
            }
    ";

    let mut frontend = naga::front::wgsl::Frontend::new();
    let module = frontend.parse(src).unwrap();

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );

    let info = validator.validate(&module).unwrap();
    // println!("info: {module:?}");
    let mut data = Vec::new();

    let mut writer = naga::back::spv::Writer::new(&Options::default()).unwrap();
    writer
        .write(
            &module,
            &info,
            Some(&PipelineOptions {
                shader_stage: naga::ShaderStage::Compute,
                entry_point: "main".into(),
            }),
            &None,
            &mut data,
        )
        .unwrap();

    let binary_slice = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, size_of_val(data.as_slice()))
    };

    println!("txt: {:?}", unsafe {
        String::from_utf8_unchecked(binary_slice.into())
    });

    let dev = CLDevice::new(0).unwrap();
    println!("dev: {:?}", dev.device.get_version());
    let devices = [dev.device.0];
    let lengths = [size_of_val(data.as_slice())];
    let binaries = [binary_slice.as_ptr()];
    let mut binary_status = 0;
    let mut errcode_ret = 0;
    let program = unsafe {
        min_cl::api::clCreateProgramWithBinary(
            dev.ctx.0,
            1,
            devices.as_ptr(),
            lengths.as_ptr(),
            binaries.as_ptr(),
            &mut binary_status,
            &mut errcode_ret,
        )
    };

    println!("binary_status: {binary_status}, errcode: {errcode_ret}, program: {program:?}");

    let program = Program(program);
    build_program(&program, &[dev.device], /*Some("-cl-std=1.2")*/ None).unwrap();
}
