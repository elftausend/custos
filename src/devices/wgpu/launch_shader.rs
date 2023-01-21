use wgpu::BindingResource;

use crate::{Buffer, Shape};

use super::WGPU;

pub trait AsBindingResource {
    fn as_binding_resource(&self) -> BindingResource;
}

impl<'a, T, S: Shape> AsBindingResource for Buffer<'a, T, WGPU, S> {
    fn as_binding_resource(&self) -> BindingResource {
        unsafe { self.ptr.buf().as_entire_binding() }
    }
}

impl<'a, T, S: Shape> AsBindingResource for &Buffer<'a, T, WGPU, S> {
    fn as_binding_resource(&self) -> BindingResource {
        unsafe { self.ptr.buf().as_entire_binding() }
    }
}

pub fn launch_shader(device: &WGPU, src: &str, gws: [u32; 3], args: &[impl AsBindingResource]) {
    let mut shader_cache = device.shader_cache.borrow_mut();
    let shader = shader_cache.shader(&device.device, src);

    let compute_pipeline =
        device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: shader,
                entry_point: "main",
            });

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);

    let bind_group_entries = args
        .iter()
        .enumerate()
        .map(|(binding, resource)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: resource.as_binding_resource(),
        })
        .collect::<Vec<wgpu::BindGroupEntry>>();

    let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &bind_group_entries,
    });

    let mut encoder = device
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("vec add");
        cpass.dispatch_workgroups(gws[0], gws[1], gws[2]);
    }

    device.queue.submit(Some(encoder.finish()));
}
