use crate::{OnDropBuffer, OpenCL, UnaryFusing};

impl<Mods: OnDropBuffer> UnaryFusing for OpenCL<Mods> {
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

            let fused_operations = operations_to_fused_src(ops);

            let src = format!(
                "
                __kernel void apply_fn(__global const {datatype}* lhs, __global {datatype}* out, long len) {{
                    size_t id = get_global_id(0);
                    if (id >= len) {{
                        return;
                    }}
                    {datatype} x = lhs[id];
                    {fused_operations}
                    out[id] = x;
                }}
            ",
                datatype = T::C_DTYPE_STR,
            );

            buf.device().launch_kernel(
                &src,
                [(buf.len() / 32 + 1) * 32, 0, 0],
                Some([32, 0, 0]),
                &[buf, *out, &buf.len()],
            )
        }
    }
}
