use super::ty::{Graphable, Type};
use crate::{
    bounds_to_range, Buffer, Device, DeviceError, Id, NoHasher, OpArgs, Parents, PtrConv, Shape,
    UniqueId,
};
use core::{any::Any, hash::BuildHasherDefault, mem::transmute, ops::RangeBounds};
use std::collections::HashMap;

pub type ForwardFn = *mut dyn Fn(&'static ()) -> crate::Result<()>;
pub type TypedForwardFn<'a, T, D, S> =
    *mut (dyn Fn(&mut Buffer<T, D, S>) -> crate::Result<()> + 'a);

pub type ForwardFn2 = *mut dyn Fn(&'static (), *mut ()) -> crate::Result<()>;

#[derive(Default)]
pub struct LazyGraph {
    pub operations: Vec<(Type, ForwardFn)>,
    pub operations2: Vec<(Type, ForwardFn2)>,
    pub args: Vec<*mut ()>,
    pub ids_to_check: Vec<Vec<UniqueId>>,
}

impl Drop for LazyGraph {
    fn drop(&mut self) {
        for (_, operation) in self.operations.iter_mut() {
            unsafe { drop(Box::from_raw(*operation)) }
        }
        for arg in self.args.iter_mut() {
            unsafe { drop(Box::from_raw(*arg)) }
        }
    }
}

pub fn execute_with_type<T: 'static, D: Device + 'static>(
    operation: &mut ForwardFn,
    buf: &mut dyn Any,
) -> crate::Result<()> {
    let operation = unsafe { transmute::<_, &mut TypedForwardFn<T, D, ()>>(operation) };

    let buf: &mut Buffer<T, D, ()> =
        unsafe { &mut *(buf as *mut dyn Any as *mut Buffer<T, D, ()>) };
    unsafe { (**operation)(buf) }
}

pub fn execute_operation<D: Device + 'static>(
    ty: Type,
    operation: &mut ForwardFn,
    buf: &mut dyn Any,
) -> crate::Result<()> {
    match ty {
        Type::F32 => execute_with_type::<f32, D>(operation, buf),
        Type::I32 => execute_with_type::<i32, D>(operation, buf),
    }
}

pub type TypedForwardFn2<'a, T, D, S, Args> =
    *mut (dyn Fn(&mut Buffer<T, D, S>, &Args) -> crate::Result<()> + 'a);

impl LazyGraph {
    pub fn add_operation_op_args<T, D, S, Args: Parents<N>, const N: usize>(
        &mut self,
        args: Args,
        operation: impl Fn(&mut Buffer<T, D, S>, &Args) -> crate::Result<()>,
    ) where
        T: Graphable,
        D: PtrConv,
        S: Shape,
    {
        let args: &mut dyn Parents<N> = Box::leak(Box::new(args));

        // store ids and test if buffers are still in cache
        self.ids_to_check
            .push(args.ids().into_iter().map(|id| *id).collect());

        self.args.push((args as *mut dyn Parents<N>).cast());

        let operation = Box::leak(Box::new(operation));
        self.operations2.push((
            T::TYPE,
            operation as TypedForwardFn2<T, D, S, Args> as *mut _,
        ))
    }

    pub unsafe fn call_lazily_op_args<D: Device + 'static>(
        &mut self,
        out_buf_order: &[Id],
        outs_unordered: &mut HashMap<UniqueId, Box<dyn Any>, BuildHasherDefault<NoHasher>>,
    ) -> crate::Result<()> {
        for ((((ty, operation), buf_id), op_arg), ids_to_check) in self
            .operations2
            .iter_mut()
            .zip(out_buf_order)
            .zip(&self.args)
            .zip(&self.ids_to_check)
        {
            for id_to_check in ids_to_check.iter() {
                outs_unordered.get(id_to_check).ok_or(DeviceError::InvalidLazyOutBuf)?;
            }

            let buf = &mut **outs_unordered
                .get_mut(buf_id)
                .ok_or(DeviceError::InvalidLazyOutBuf)?;

            match ty {
                Type::F32 => {
                    let operation =
                        transmute::<_, &mut *mut dyn Fn(&Buffer<f32, D, ()>, &())>(operation);

                    let buf: &mut Buffer<f32, D, ()> =
                        unsafe { &mut *(buf as *mut dyn Any as *mut Buffer<f32, D, ()>) };

                    unsafe { (**operation)(buf, &**op_arg) }
                }
                Type::I32 => todo!(),
            }
            // execute_operation::<D>(*ty, operation, buf)?;
        }
        Ok(())
    }

    pub fn add_operation<T, D, S>(
        &mut self,
        operation: impl Fn(&mut Buffer<T, D, S>) -> crate::Result<()>,
    ) where
        T: Graphable,
        D: PtrConv,
        S: Shape,
    {
        let operation = Box::leak(Box::new(operation));
        self.operations
            .push((T::TYPE, operation as TypedForwardFn<T, D, S> as *mut _))
    }

    /// # Safety
    /// The required 'static lifetime is ignored when adding operations. Hence, all captured variables must live long enough.
    pub unsafe fn call_lazily<D: Device + 'static>(
        &mut self,
        out_buf_order: &[Id],
        outs_unordered: &mut HashMap<UniqueId, Box<dyn Any>, BuildHasherDefault<NoHasher>>,
    ) -> crate::Result<()> {
        for ((ty, operation), buf_id) in self.operations.iter_mut().zip(out_buf_order) {
            let buf = &mut **outs_unordered
                .get_mut(buf_id)
                .ok_or(DeviceError::InvalidLazyOutBuf)?;

            execute_operation::<D>(*ty, operation, buf)?;
        }
        Ok(())
    }

    pub unsafe fn call_range<D: Device + 'static>(
        &mut self,
        bounds: impl RangeBounds<usize>,
        out_buf_order: &mut Vec<Id>,
        outs_unordered: &mut HashMap<UniqueId, Box<dyn Any>, BuildHasherDefault<NoHasher>>,
    ) -> crate::Result<()> {
        let range = bounds_to_range(bounds, out_buf_order.len());
        for ((ty, mut operation), buf_id) in self
            .operations
            .drain(range.clone())
            .zip(out_buf_order.drain(range))
        {
            let buf = &mut **outs_unordered
                .get_mut(&buf_id)
                .ok_or(DeviceError::InvalidLazyOutBuf)?;

            execute_operation::<D>(ty, &mut operation, buf)?;
            unsafe { drop(Box::from_raw(operation)) }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{register_buf, Base, Buffer, Device, HasId, Retriever, CPU};

    use super::LazyGraph;

    #[test]
    fn test_lazy_op_args() {
        let device = CPU::<Base>::new();
        let mut graph = LazyGraph::default();

        let lhs = device.buffer([1f32, 2., 3., 4., 5.]);
        let rhs = device.buffer([1f32, 2., 6., 4., 5.]);

        let mut outs_unordered = HashMap::default();

        let out: Buffer = device.retrieve(lhs.len(), (&lhs, &rhs));
        unsafe { register_buf(&mut outs_unordered, &lhs) };
        unsafe { register_buf(&mut outs_unordered, &rhs) };
        unsafe { register_buf(&mut outs_unordered, &out) };
        // outs_unordered.insert(out.id(), )

        graph.add_operation_op_args::<f32, CPU, (), _, 2>((&lhs, &rhs), |out, args| {
            let (lhs, rhs) = *args;
            println!("args: {lhs:?}");
            Ok(())
        });

        unsafe {
            graph.call_lazily_op_args::<CPU>(&[out.id()], &mut outs_unordered).unwrap()
        }
    }
}
