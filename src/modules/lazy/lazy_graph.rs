use super::ty::{Graphable, Type};
use crate::{Buffer, Device, DeviceError, Id, NoHasher, PtrConv, Shape, UniqueId};
use core::{any::Any, hash::BuildHasherDefault, mem::transmute};
use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct LazyGraph {
    operations: Vec<(Type, *mut dyn Fn(&'static ()))>,
}

impl Drop for LazyGraph {
    fn drop(&mut self) {
        for (_, operation) in self.operations.iter_mut() {
            unsafe { drop(Box::from_raw(*operation)) }
        }
    }
}

impl LazyGraph {
    pub fn add_operation<T, D, S>(&mut self, operation: impl Fn(&mut Buffer<T, D, S>))
    where
        T: Graphable,
        D: PtrConv,
        S: Shape,
    {
        let operation = Box::leak(Box::new(operation));
        self.operations.push((
            T::TYPE,
            operation as *mut dyn Fn(&mut Buffer<T, D, S>) as *mut _,
        ))
    }

    pub fn call_lazily<D: Device>(
        &mut self,
        out_buf_order: &[Id],
        outs_unordered: &mut HashMap<UniqueId, Box<dyn Any>, BuildHasherDefault<NoHasher>>,
    ) -> Result<(), DeviceError> {
        for ((ty, operation), buf_id) in self.operations.iter_mut().zip(out_buf_order) {
            let buf = &mut **outs_unordered
                .get_mut(buf_id)
                .ok_or(DeviceError::InvalidLazyOutBuf)?;
            match ty {
                Type::F32 => {
                    let operation = unsafe {
                        transmute::<_, &mut *mut dyn Fn(&mut Buffer<f32, D, ()>)>(operation)
                    };

                    let buf: &mut Buffer<f32, D, ()> = unsafe { &mut *(buf as *mut _ as *mut _) };
                    unsafe {
                        (**operation)(buf);
                    }
                }
                Type::I32 => {
                    let operation = unsafe {
                        transmute::<_, &mut *mut dyn Fn(&mut Buffer<i32, D, ()>)>(operation)
                    };

                    let buf: &mut Buffer<i32, D, ()> = unsafe { &mut *(buf as *mut _ as *mut _) };
                    unsafe {
                        (**operation)(buf);
                    }
                }
            }
        }
        Ok(())
    }
}
