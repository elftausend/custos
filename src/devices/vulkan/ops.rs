use crate::{
    cpu_stack_ops::clear_slice, pass_down_add_operation, Buffer, CDatatype, ClearBuf, OnDropBuffer,
    Read, Shape, UseGpuOrCpu, Vulkan,
};

use super::VkArray;

pass_down_add_operation!(Vulkan);

impl<Mods: OnDropBuffer + UseGpuOrCpu, T: CDatatype + Default> ClearBuf<T> for Vulkan<Mods> {
    #[inline]
    fn clear(&self, buf: &mut Buffer<T, Vulkan<Mods>>) {
        let mut cpu_buf = unsafe { &mut *(buf as *mut Buffer<_, _, _>) };
        let info = self.use_cpu_or_gpu(
            (file!(), line!(), column!()).into(),
            &[buf.len()],
            || clear_slice(&mut cpu_buf),
            || try_vk_clear(self, &mut buf.data).unwrap(),
        );
        println!("info: {info:?}")
    }
}

pub fn try_vk_clear<Mods, T>(device: &Vulkan<Mods>, buf: &mut VkArray<T>) -> crate::Result<()> {
    let src = format!(
        "@group(0)
            @binding(0)
            var<storage, read_write> buf: array<{dtype}>;
            
            @compute
            @workgroup_size(32)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                if global_id.x >= arrayLength(&buf) {{
                    return;    
                }}
                buf[global_id.x] = 0; 
            }}
    ",
        dtype = std::any::type_name::<T>()
    );

    device.launch_shader([(32 + buf.len as u32) / 32, 1, 1], src, &[buf])
}

impl<Mods: OnDropBuffer, T, S: Shape> Read<T, S> for Vulkan<Mods> {
    type Read<'a> = &'a [T]
    where
        T: 'a,
        Self: 'a,
        S: 'a;

    #[inline]
    fn read<'a>(&self, buf: &'a Buffer<T, Self, S>) -> Self::Read<'a> {
        buf.as_slice()
    }

    #[inline]
    fn read_to_vec(&self, buf: &Buffer<T, Self, S>) -> Vec<T>
    where
        T: Default + Clone,
    {
        buf.as_slice().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use crate::{Base, Buffer, Fork, Vulkan};

    use super::try_vk_clear;

    #[test]
    fn test_try_vk_clear() {
        let device = Vulkan::<Base>::new(0).unwrap();
        let mut buf = Buffer::from((&device, [1f32, 2., 3., 4., 5., 6.]));

        try_vk_clear(&device, &mut buf.data).unwrap();
        assert_eq!(buf.read(), [0f32; 6])
    }

    #[test]
    fn test_vk_inplace_clear() {
        let device = Vulkan::<Base>::new(0).unwrap();
        let mut buf = Buffer::from((&device, [1f32, 2., 3., 4., 5., 6.]));
        buf.clear();
        assert_eq!(buf.read(), [0f32; 6])
    }

    #[cfg(feature = "fork")]
    #[test]
    fn test_vk_inplace_clear_fork() {
        let device = Vulkan::<Fork<Base>>::new(0).unwrap();
        let mut buf = Buffer::from((&device, [1f32, 2., 3., 4., 5., 6.]));
        buf.clear();
        assert_eq!(buf.read(), [0f32; 6])
    }
    
    #[cfg(feature = "fork")]
    #[ignore = "to long runtime"]
    #[test]
    fn test_vk_inplace_clear_fork_multiple_times() {
        let device = Vulkan::<Fork<Base>>::new(1).unwrap();
        let mut buf = Buffer::from((&device, vec![1; 10000000]));
        buf.clear();
        assert_eq!(buf.read(), vec![0; 10000000])
    }
}
