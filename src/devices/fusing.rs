use crate::IsShapeIndep;

#[cfg(feature = "std")]
pub fn operations_to_fused_src<T: Default + Copy>(
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

pub trait UnaryFusing: IsShapeIndep {
    #[cfg(feature = "lazy")]
    #[cfg(feature = "graph")]
    fn unary_fuse_op<T: crate::CDatatype + crate::Numeric>(
        &self,
    ) -> fn(
        &mut (
            &mut crate::Buffer<'_, T, Self, ()>,
            &crate::Buffer<'_, T, Self, ()>,
            crate::NoId<Vec<std::rc::Rc<dyn Fn(crate::Resolve<T>) -> Box<dyn crate::TwoWay<T>>>>>,
        ),
    ) -> crate::Result<()>;

    #[cfg(feature = "lazy")]
    #[cfg(feature = "graph")]
    /// # Safety
    /// Does not check if specific retrieved buffers contain data of type `T`.
    unsafe fn fuse_unary_ops<T: crate::CDatatype + crate::Numeric>(
        &self,
        lazy_graph: &crate::LazyGraph<Box<dyn crate::BoxedShallowCopy>, T>,
        ops: (
            Vec<std::rc::Rc<dyn Fn(crate::Resolve<T>) -> Box<dyn crate::TwoWay<T>>>>,
            Vec<usize>,
        ),
        buffers: &mut crate::Buffers<Box<dyn crate::BoxedShallowCopy>>,
    ) -> (usize, crate::Operation<Box<dyn crate::BoxedShallowCopy>, T>)
    where
        Self: 'static,
    {
        use crate::{AsAny, AsNoId, Buffer};

        let (ops, affected_op_idxs) = ops;
        let to_insert_idx: usize = affected_op_idxs[0];

        let first_op = &lazy_graph.operations[to_insert_idx];

        let first_arg_ids = first_op
            .arg_ids
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();

        let last_op = &lazy_graph.operations[*affected_op_idxs.last().unwrap()];

        // use last op in the unary fuse chain as the output buffer
        let last_arg_ids = last_op
            .arg_ids
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();

        let out = unsafe {
            &mut *(buffers.get_mut(&last_arg_ids[0]).unwrap().as_any_mut()
                as *mut Buffer<T, Self, ()>)
        };

        let buf = unsafe {
            &*(buffers.get(&first_arg_ids[1]).unwrap().as_any() as *const Buffer<T, Self, ()>)
        };

        let op = self.unary_fuse_op::<T>();
        let mut operation =
            unsafe { crate::LazyGraph::convert_to_operation((out, buf, ops.no_id()), op) };
        // using the buffers out of the 'buffers' hashmaps results in using allocated buffers that are not in the 'buffers' hashmap
        // if the lazy graph is executed, it updates the references to the corresponding buffers -> new ids would not be found -> invalid lazy buffer panic
        operation.arg_ids = vec![Some(last_arg_ids[0]), Some(first_arg_ids[1]), None];
        operation.op_hint = crate::op_hint::OpHint::UnaryFused;

        (to_insert_idx, operation)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "std")]
    #[test]
    fn test_operations_to_fused_src() {
        use crate::{
            op_hint::{unary, OpHint},
            operations_to_fused_src, Combiner, Resolve,
        };

        let mut ops = vec![];
        ops.push(unary(|x: Resolve<f32>| x.sin()));
        ops.push(unary(|x: Resolve<f32>| x.neg()));
        ops.push(unary(|x: Resolve<f32>| x.cos()));

        let ops = ops
            .into_iter()
            .map(|op| {
                let OpHint::Unary(op) = op else { panic!() };
                op
            })
            .collect::<Vec<_>>();

        let src = operations_to_fused_src(&ops);
        assert_eq!(src, "x = sin(x);\nx = -(x);\nx = cos(x);\n")
    }
}
