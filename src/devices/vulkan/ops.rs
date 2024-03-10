use core::fmt::Debug;

use crate::{
    cpu_stack_ops::clear_slice, pass_down_add_operation, pass_down_exec_now, prelude::Number,
    AddOperation, ApplyFunction, AsNoId, BufAsNoId, Buffer, CDatatype, ClearBuf, HostPtr,
    OnDropBuffer, Read, Resolve, Retrieve, Retriever, Shape, ToMarker, ToWgslSource, UnaryGrad,
    UseGpuOrCpu, Vulkan, WriteBuf, ZeroGrad,
};

use super::{VkArray, VkDevice};

pass_down_add_operation!(Vulkan);
pass_down_exec_now!(Vulkan);

impl<Mods: OnDropBuffer + UseGpuOrCpu, T: CDatatype + Default + Debug> ClearBuf<T>
    for Vulkan<Mods>
{
    #[inline]
    fn clear(&self, buf: &mut Buffer<T, Vulkan<Mods>>) {
        let cpu_buf = unsafe { &mut *(buf as *mut Buffer<T, Vulkan<Mods>>) };
        self.use_cpu_or_gpu(
            (file!(), line!(), column!()).into(),
            &[buf.len()],
            || clear_slice(cpu_buf),
            || try_vk_clear(self, buf).unwrap(),
        );
    }
}

impl<Mods: OnDropBuffer, T: Default + Debug> ZeroGrad<T> for Vulkan<Mods> {
    #[inline]
    fn zero_grad<S: Shape>(&self, data: &mut Self::Base<T, S>) {
        try_vk_clear(self, data).unwrap()
    }
}

pub fn try_vk_clear<T: Default + Debug>(
    device: &VkDevice,
    buf: &mut VkArray<T>,
) -> crate::Result<()> {
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
            buf[global_id.x] = {default:?}; 
        }}
    ",
        dtype = std::any::type_name::<T>(),
        default = T::default(),
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
    Mods: AddOperation + Retrieve<Self, T, S> + UseGpuOrCpu + 'static,
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

        // self.add_op(&mut out, move |out| {
        let cpu_out = unsafe { &mut *(&mut out as *mut Buffer<T, Vulkan<Mods>, _>) };
        self.use_cpu_or_gpu(
            (file!(), line!(), column!()).into(),
            &[buf.len()],
            || crate::devices::cpu_stack_ops::apply_fn_slice(buf, cpu_out, f),
            || try_vk_apply_fn_mut(self, &buf, &mut out, f).unwrap(),
        );
        // Ok(())
        // })
        // .unwrap();

        out
    }
}

pub fn try_vk_apply_fn_mut<T, F>(
    device: &VkDevice,
    x: &VkArray<T>,
    out: &mut VkArray<T>,
    f: impl Fn(Resolve<T>) -> F,
) -> crate::Result<()>
where
    T: Number,
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

impl<T, S, Mods> UnaryGrad<T, S> for Vulkan<Mods>
where
    T: CDatatype + Number,
    S: Shape,
    Mods: OnDropBuffer + AddOperation + 'static,
{
    #[inline]
    fn add_unary_grad<F>(
        &self,
        lhs: &Buffer<T, Self, S>,
        lhs_grad: &mut Buffer<T, Self, S>,
        out: &Buffer<T, Self, S>,
        lhs_grad_fn: impl Fn(Resolve<T>) -> F + Copy + 'static,
    ) where
        F: ToWgslSource,
    {
        self.add_op::<_, 4>(
            (lhs, lhs_grad.buf_no_id(), out, lhs_grad_fn.no_id()),
            move |(lhs, lhs_grad, out, lhs_grad_fn)| {
                try_vk_add_unary_grad(lhs.device(), lhs, lhs_grad, out, **lhs_grad_fn)
            },
        )
        .unwrap();
    }
}
pub fn try_vk_add_unary_grad<T, F>(
    device: &VkDevice,
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

impl<Mods: OnDropBuffer, T: Clone, S: Shape> WriteBuf<T, S> for Vulkan<Mods> {
    #[inline]
    fn write(&self, buf: &mut Buffer<T, Self, S>, data: &[T]) {
        buf.as_mut_slice().clone_from_slice(data)
    }

    #[inline]
    fn write_buf(&self, dst: &mut Buffer<T, Self, S>, src: &Buffer<T, Self, S>) {
        dst.write(src);
    }
}

#[cfg(test)]
mod tests {
    use super::{try_vk_apply_fn_mut, try_vk_clear};
    use crate::{vulkan::ops::try_vk_add_unary_grad, Base, Buffer, Combiner, Vulkan};

    #[cfg(feature = "fork")]
    use crate::Fork;

    #[test]
    fn test_try_vk_clear() {
        let device = Vulkan::<Base>::new(0).unwrap();
        let mut buf = Buffer::from((&device, [1f32, 2., 3., 4., 5., 6.]));

        try_vk_clear(&device, &mut buf.data).unwrap();
        assert_eq!(buf.read(), [0f32; 6]);
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
