use super::ty::{Graphable, Type};
use crate::{bounds_to_range, Buffer, Device, DeviceError, Id, NoHasher, PtrConv, Shape, UniqueId};
use core::{any::Any, hash::BuildHasherDefault, mem::transmute, ops::RangeBounds};
use std::collections::HashMap;

pub type ForwardFn = *mut dyn Fn(&'static ()) -> crate::Result<()>;
pub type TypedForwardFn<'a, T, D, S> =
    *mut (dyn Fn(&mut Buffer<T, D, S>) -> crate::Result<()> + 'a);

#[derive(Debug, Default)]
pub struct LazyGraph {
    pub operations: Vec<(Type, ForwardFn)>,
}

impl Drop for LazyGraph {
    fn drop(&mut self) {
        for (_, operation) in self.operations.iter_mut() {
            unsafe { drop(Box::from_raw(*operation)) }
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

impl LazyGraph {
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
        for ((ty, mut operation), buf_id) in self.operations.drain(range.clone())
            .zip(out_buf_order.drain(range))
        {
            let buf = &mut **outs_unordered
                .get_mut(&buf_id)
                .ok_or(DeviceError::InvalidLazyOutBuf)?;

            execute_operation::<D>(ty, &mut operation, buf)?;
            unsafe {
                drop(Box::from_raw(operation))
            }
        }
        Ok(())
    }
}
