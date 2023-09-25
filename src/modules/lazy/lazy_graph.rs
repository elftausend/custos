use core::mem::transmute;
use crate::{Device, PtrConv, Shape, Buffer};
use super::ty::{Graphable, Type};

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
        self.operations
            .push((T::TYPE, operation as *mut dyn Fn(&mut _) as *mut _))
    }

    pub fn call_lazily<D: Device>(&mut self) {
        for (ty, operation) in self.operations.iter_mut() {
            match ty {
                Type::F32 => {
                    let operation = unsafe {
                        transmute::<_, &mut *mut dyn Fn(&mut Buffer<f32, D, ()>)>(operation)
                    };
                    // let mut out = vec![0f32; 100];
                    // unsafe {
                    //     (**operation)(&mut out);
                    // }
                }
                Type::I32 => {
                    let operation = unsafe {
                        transmute::<_, &mut *mut dyn Fn(&mut Buffer<i32, D, ()>)>(operation)
                    };

                    // let mut out = vec![0i32; 100];
                    // unsafe {
                    //     (**operation)(&mut out);
                    // }
                }
            }
        }
    }
}
