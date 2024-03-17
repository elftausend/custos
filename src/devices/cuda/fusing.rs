use crate::{OnDropBuffer, UnaryFusing, CUDA};

impl<Mods: OnDropBuffer> UnaryFusing for CUDA<Mods> {
    #[cfg(feature = "lazy")]
    #[cfg(feature = "graph")]
    #[inline]
    fn unary_fuse_op<T: crate::CDatatype + crate::Numeric>(
        &self,
    ) -> fn(
        &mut (
            &mut crate::Buffer<'_, T, Self, ()>,
            &crate::Buffer<'_, T, Self, ()>,
            crate::NoId<Vec<std::rc::Rc<dyn Fn(crate::Resolve<T>) -> Box<dyn crate::TwoWay<T>>>>>,
        ),
    ) -> crate::Result<()> {
        use crate::operations_to_fused_src;
        |(out, buf, ops)| {
            if ops.is_empty() {
                return Ok(());
            }

            let fused_operations = operations_to_fused_src(&ops);

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
                &[buf, *out, &buf.len()],
            )
        }
    }
}
