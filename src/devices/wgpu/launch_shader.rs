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

impl<'a, T, S: Shape> AsBindingResource for &mut Buffer<'a, T, WGPU, S> {
    fn as_binding_resource(&self) -> BindingResource {
        unsafe { self.ptr.buf().as_entire_binding() }
    }
}

/// Launches a `WGPU` compute shader.
///
/// # Example
///
/// ```
/// use custos::{Buffer, Shape, number::Number, WGPU, wgpu::launch_shader};
///
/// pub fn wgpu_element_wise<T: Number, S: Shape>(
///     device: &WGPU,
///     lhs: &Buffer<T, WGPU, S>,
///     rhs: &Buffer<T, WGPU, S>,
///     out: &mut Buffer<T, WGPU, S>,
///     op: &str,
/// ) {
///     let src = format!(
///         "@group(0)
///         @binding(0)
///         var<storage, read_write> lhs: array<{datatype}>;
///         
///         @group(0)
///         @binding(1)
///         var<storage, read_write> rhs: array<{datatype}>;
///     
///         @group(0)
///         @binding(2)
///         var<storage, read_write> out: array<{datatype}>;
///         
///         
///         @compute
///         @workgroup_size(1)
///         fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
///             out[global_id.x] = lhs[global_id.x] {op} rhs[global_id.x];
///         }}
///         ",
///         datatype = std::any::type_name::<T>()
///     );
///
///     launch_shader(device, &src, [lhs.len() as u32, 1, 1], &[lhs, rhs, out]);
/// }
///
/// fn main() -> custos::Result<()> {
///     let device = WGPU::new(wgpu::Backends::all())?;
///     
///     let lhs = Buffer::from((&device, [1, 2, 3, 4, 5]));
///     let rhs = Buffer::from((&device, [-1, 6, 4, -2, 3]));
///
///     let mut out = Buffer::new(&device, 5);
///
///     wgpu_element_wise(&device, &lhs, &rhs, &mut out, "+");
///
///     assert_eq!(out.read(), [0, 8, 7, 2, 8]);
///
///     Ok(())
/// }
///
/// ```
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
