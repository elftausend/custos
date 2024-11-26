use crate::{UnaryFusing, WrappedData, CUDA};

impl<Mods: WrappedData> UnaryFusing for CUDA<Mods> {
    #[cfg(feature = "lazy")]
    #[cfg(feature = "graph")]
    #[inline]
    fn unary_fuse_op<T: crate::CDatatype + crate::Numeric>(
        &self,
        ops_to_fuse: Vec<std::rc::Rc<dyn Fn(crate::Resolve<T>) -> Box<dyn crate::TwoWay<T>>>>,
    ) -> Box<
        dyn Fn(
            (
                &mut crate::Buffer<'_, T, Self, ()>,
                &crate::Buffer<'_, T, Self, ()>,
            ),
        ) -> crate::Result<()>,
    > {
        use crate::operations_to_fused_src;
        Box::new(move |(out, buf)| {
            if ops_to_fuse.is_empty() {
                return Ok(());
            }

            let fused_operations = operations_to_fused_src(&ops_to_fuse);

            let src = format!(
                r#"extern "C" __global__ void applyFn({datatype}* lhs, {datatype}* out, int numElements)
                    {{
                        int idx = blockDim.x * blockIdx.x + threadIdx.x;
                        if (idx >= numElements) {{
                            return;
                        }}
                        {datatype} x = lhs[idx];
                        {fused_operations}
                        out[idx] = x;
                    }}
                "#,
                datatype = T::C_DTYPE_STR
            );
            buf.device().launch_kernel(
                &src,
                "applyFn",
                [(buf.len() as u32 / 32 + 1) * 32, 1, 1],
                [32, 1, 1],
                0,
                &[buf, out, &buf.len()],
            )
        })
    }
}
