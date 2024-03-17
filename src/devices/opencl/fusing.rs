use crate::{OnDropBuffer, OpenCL, UnaryFusing};

#[cfg(feature = "std")]
fn operations_to_fused_src<T: Default + Copy>(
    ops: &[std::rc::Rc<dyn Fn(crate::Resolve<T>) -> Box<dyn crate::TwoWay<T>>>],
) -> String {
    ops.iter().fold(String::new(), |acc, op| {
        let resolve = crate::Resolve {
            val: T::default(),
            marker: "x",
        };

        format!(
            "{acc}{marker} = {src};\n",
            marker = resolve.marker,
            src = op(resolve).to_cl_source()
        )
    })
}

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
        |(out, buf, ops)| {
            if ops.is_empty() {
                return Ok(());
            }

            let fused_operations = operations_to_fused_src(&ops);

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

#[cfg(test)]
mod tests {
    #[cfg(feature = "std")]
    #[test]
    fn test_operations_to_fused_src() {
        use crate::{
            op_hint::{unary, OpHint},
            opencl::fusing::operations_to_fused_src,
            Combiner, Resolve,
        };

        let mut ops = vec![];
        ops.push(unary(|x: Resolve<f32>| x.sin()));
        ops.push(unary(|x: Resolve<f32>| x.neg()));
        ops.push(unary(|x: Resolve<f32>| x.cos()));

        let ops = ops.into_iter().map(|op| {
            let OpHint::Unary(op) = op else { panic!() };
            op
        }).collect::<Vec<_>>();

        let src = operations_to_fused_src(&ops);
        assert_eq!(src, "x = sin(x);\nx = -(x);\nx = cos(x);\n")
    }
}
