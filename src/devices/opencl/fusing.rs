use crate::{OpenCL, UnaryFusing, WrappedData};

impl<Mods: WrappedData> UnaryFusing for OpenCL<Mods> {
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
            ), &Self,
        ) -> crate::Result<()>,
    > {
        use crate::operations_to_fused_src;

        Box::new(move |(out, buf), _| {
            if ops_to_fuse.is_empty() {
                return Ok(());
            }

            let fused_operations = operations_to_fused_src(&ops_to_fuse);

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
                &[buf, out, &buf.len()],
            )
        })
    }
}
