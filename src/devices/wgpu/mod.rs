mod launch_shader;
mod shader_cache;
mod wgpu_buffer;
pub mod wgpu_device;

use core::fmt::Debug;

pub use launch_shader::*;
pub use wgpu_device::*;

use crate::{Buffer, Shape};

/// Sets all the elements of a `WGPU` `Buffer` to zero / default.
/// 
/// # Example
/// ```
/// use crate::{WGPU, Buffer, wgpu::wgpu_clear};
/// 
/// fn main() -> custos::Result<()> {
///     let device = WGPU::new(wgpu::Backends::all())?;
///     let mut buf = Buffer::from((&device, [4f32, -1.2, 2., 1., 3.]));
///     wgpu_clear(&device, &mut buf);
///     
///     assert_eq!(buf.read(), [0., 0., 0., 0., 0.,]);
///     Ok(())
/// }
/// ```
pub fn wgpu_clear<T: Default + Debug, S: Shape>(device: &WGPU, buf: &mut Buffer<T, WGPU, S>) {
    let src = format!(
        "@group(0)
        @binding(0)
        var<storage, read_write> buf: array<{datatype}>;
        
        @compute
        @workgroup_size(1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            buf[global_id.x] = {zero:?};
        }}
        ",
        datatype = std::any::type_name::<T>(),
        zero = T::default()
    );

    launch_shader(device, &src, [buf.len() as u32, 1, 1], &[buf])
}

#[cfg(test)]
mod tests {
    use crate::{WGPU, Buffer};

    #[test]
    fn test_wgpu_clear() -> crate::Result<() >{
        let device = WGPU::new(wgpu::Backends::all())?;

        let mut buf = Buffer::from((&device, [1, 4, 2, 1, 9]));
        buf.clear();

        assert_eq!(buf.read(), [0, 0, 0, 0, 0]);

        Ok(())
    }
}