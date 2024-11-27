use crate::{
    AddOperation, Alloc, ApplyFunction, OnDropBuffer, Read, Retrieve, Retriever, SetOpHint, Shape,
    ToMarker, Unit, op_hint::unary,
};

use super::{AsShaderArg, WgslShaderLaunch, wgsl_device::Wgsl};

impl<T: Unit, S: Shape, D: Read<T, S>, Mods: OnDropBuffer + 'static> Read<T, S> for Wgsl<D, Mods> {
    type Read<'a>
        = D::Read<'a>
    where
        T: 'a,
        D: 'a,
        S: 'a;

    #[inline]
    fn read<'a>(&self, buf: &'a Self::Base<T, S>) -> Self::Read<'a>
    where
        Self: 'a,
    {
        self.backend.read(buf)
    }

    #[inline]
    fn read_to_vec(&self, buf: &Self::Base<T, S>) -> Vec<T>
    where
        T: Default + Clone,
    {
        self.backend.read_to_vec(buf)
    }
}

impl<D, Mods, T, S> ApplyFunction<T, S, Self> for Wgsl<D, Mods>
where
    T: Unit + Default + 'static,
    D: WgslShaderLaunch + Alloc<T> + 'static,
    D::Base<T, S>: AsShaderArg<D>,
    Mods: SetOpHint<T> + Retrieve<Self, T, S> + AddOperation + 'static,
    S: Shape,
{
    fn apply_fn<F>(
        &self,
        buf: &crate::Buffer<T, Self, S>,
        f: impl Fn(crate::Resolve<T>) -> F + Copy + 'static,
    ) -> crate::Buffer<T, Self, S>
    where
        F: crate::TwoWay<T> + 'static,
    {
        let mut out = self.retrieve(buf.len(), buf).unwrap();

        self.add_op((&mut out, buf), move |(out, buf)| {
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

            out.device()
                .launch_shader(src, [(32 + buf.len() as u32) / 32, 1, 1], &[
                    buf.arg(),
                    out.arg_mut(),
                ])
        })
        .unwrap();
        self.modules.set_op_hint(unary(f));

        out
    }
}

#[cfg(test)]
mod tests {
    use crate::{ApplyFunction, Combiner, Device, Vulkan, wgsl::wgsl_device::Wgsl};

    #[test]
    fn test_wgsl_device_apply_fn() {
        let dev = Wgsl::<Vulkan>::new(0).unwrap();
        let x = dev.buffer([1, 2, 3]);

        let out = dev.apply_fn(&x, |x| x.add(5));
        assert_eq!(out.read_to_vec(), [6, 7, 8])
    }
}
