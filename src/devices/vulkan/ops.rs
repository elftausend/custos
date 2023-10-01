use crate::{
    cpu_stack_ops::clear_slice, pass_down_add_operation, prelude::Number, AddOperation,
    ApplyFunction, Buffer, CDatatype, ClearBuf, OnDropBuffer, Read, Resolve, Retrieve, Retriever,
    Shape, ToMarker, ToWgslSource, UnaryGrad, UseGpuOrCpu, Vulkan,
};

use super::VkArray;

pass_down_add_operation!(Vulkan);

impl<Mods: OnDropBuffer + UseGpuOrCpu, T: CDatatype + Default> ClearBuf<T> for Vulkan<Mods> {
    #[inline]
    fn clear(&self, buf: &mut Buffer<T, Vulkan<Mods>>) {
        let cpu_buf = unsafe { &mut *(buf as *mut Buffer<_, _, _>) };
        self.use_cpu_or_gpu(
            (file!(), line!(), column!()).into(),
            &[buf.len()],
            || clear_slice(cpu_buf),
            || try_vk_clear(self, &mut buf.data).unwrap(),
        );
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

impl<Mods, T, S> ApplyFunction<T, S> for Vulkan<Mods>
where
    T: Number,
    Mods: AddOperation<T, Self> + Retrieve<Self, T> + UseGpuOrCpu + 'static,
    S: Shape,
{
    #[inline]
    fn apply_fn<F>(
        &self,
        buf: &Buffer<T, Self, S>,
        f: impl Fn(Resolve<T>) -> F + Copy,
    ) -> Buffer<T, Self, S>
    where
        F: crate::Eval<T> + crate::MayToWgslSource,
    {
        let mut out = self.retrieve(buf.len(), buf);

        self.add_op(&mut out, |out| {
            #[cfg(unified_cl)]
            {
                let cpu_out = unsafe { &mut *(out as *mut Buffer<_, _, _>) };
                self.use_cpu_or_gpu(
                    (file!(), line!(), column!()).into(),
                    &[buf.len()],
                    || crate::devices::cpu_stack_ops::apply_fn_slice(buf, cpu_out, f),
                    || try_vk_apply_fn_mut(self, &buf.data, &mut out.data, f).unwrap(),
                );
            }
            #[cfg(not(unified_cl))]
            try_cl_apply_fn_mut(self, buf, &mut out, f).unwrap();
        });

        out
    }
}

pub fn try_vk_apply_fn_mut<T, Mods, F>(
    device: &Vulkan<Mods>,
    x: &VkArray<T>,
    out: &mut VkArray<T>,
    f: impl Fn(Resolve<T>) -> F,
) -> crate::Result<()>
where
    T: Number,
    Mods: OnDropBuffer,
    F: ToWgslSource,
{
    let src = format!(
        "
        @group(0)
        @binding(0)
        var<storage, read_write> x: array<{dtype}>;

        @group(0)
        @binding(1)
        var<storage, read_write> out: array<{dtype}>;
        
        @compute
        @workgroup_size(32)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            if global_id.x >= arrayLength(&out) {{
                return;    
            }}
            out[global_id.x] = {op};
        }}

    ",
        dtype = std::any::type_name::<T>(),
        op = f("x[global_id.x]".to_marker()).to_wgsl_source()
    );
    device.launch_shader([(32 + x.len as u32) / 32, 1, 1], src, &[x, out])
}

impl<T, S, Mods: OnDropBuffer + AddOperation<T, Self>> UnaryGrad<T, S> for Vulkan<Mods>
where
    T: CDatatype + Number,
    S: Shape,
{
    #[inline]
    fn add_unary_grad<F>(
        &self,
        lhs: &Buffer<T, Self, S>,
        lhs_grad: &mut Buffer<T, Self, S>,
        out: &Buffer<T, Self, S>,
        lhs_grad_fn: impl Fn(Resolve<T>) -> F + Copy,
    ) where
        F: ToWgslSource,
    {
        self.add_op(lhs_grad, |lhs_grad| {
            try_vk_add_unary_grad(self, &lhs.data, &mut lhs_grad.data, &out.data, lhs_grad_fn)
                .unwrap()
        });
    }
}
pub fn try_vk_add_unary_grad<T, F, Mods: OnDropBuffer>(
    device: &Vulkan<Mods>,
    lhs: &VkArray<T>,
    lhs_grad: &mut VkArray<T>,
    out: &VkArray<T>,
    lhs_grad_fn: impl Fn(Resolve<T>) -> F,
) -> crate::Result<()>
where
    T: CDatatype + Number,
    F: ToWgslSource,
{
    let src = format!(
        "
        @group(0)
        @binding(0)
        var<storage, read_write> lhs: array<{dtype}>;

        @group(0)
        @binding(1)
        var<storage, read_write> lhs_grad: array<{dtype}>;
        
        @group(0)
        @binding(2)
        var<storage, read_write> out: array<{dtype}>;
        
        @compute
        @workgroup_size(32)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            if global_id.x >= arrayLength(&out) {{
                return;    
            }}
            lhs_grad[global_id.x] += out[global_id.x] * {op};
        }}

    ",
        dtype = std::any::type_name::<T>(),
        op = lhs_grad_fn("lhs[global_id.x]".to_marker()).to_wgsl_source()
    );
    device.launch_shader(
        [(32 + lhs.len as u32) / 32, 1, 1],
        src,
        &[lhs, lhs_grad, out],
    )
}
#[cfg(test)]
mod tests {
    use crate::{vulkan::ops::try_vk_add_unary_grad, Base, Buffer, Combiner, Fork, Vulkan};

    use super::{try_vk_apply_fn_mut, try_vk_clear};

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

    #[test]
    fn test_vk_apply_fn() {
        let device = Vulkan::<Base>::new(0).unwrap();
        let x = Buffer::from((&device, [1f32, 2., 3., 4., 5., 6.]));
        let mut out = x.empty_like();
        try_vk_apply_fn_mut(&device, &x.data, &mut out.data, |x| x.add("1.0")).unwrap();
        assert_eq!(out.read(), [2f32, 3., 4., 5., 6., 7.,])
    }

    #[test]
    #[ignore = "f64 currenlty not supported"]
    fn test_vk_apply_fn_f64() {
        let device = Vulkan::<Base>::new(0).unwrap();
        let x = Buffer::from((&device, [1f64, 2., 3., 4., 5., 6.]));
        let mut out = x.empty_like();
        try_vk_apply_fn_mut(&device, &x.data, &mut out.data, |x| x.add(1f64)).unwrap();
        assert_eq!(out.read(), [2f64, 3., 4., 5., 6., 7.,])
    }

    #[test]
    fn test_vk_add_unary_grad() -> crate::Result<()> {
        let device = Vulkan::<Base>::new(0)?;
        let lhs = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));
        let mut lhs_grad = Buffer::from((&device, [1, 2, 3, 4, 5, 6]));

        let out = Buffer::from((&device, [1, 1, 1, 1, 1, 1]));

        try_vk_add_unary_grad(&device, &lhs.data, &mut lhs_grad.data, &out.data, |x| {
            x.mul(2).add(1)
        })?;

        assert_eq!(lhs_grad.read(), [4, 7, 10, 13, 16, 19]);

        Ok(())
    }
}
