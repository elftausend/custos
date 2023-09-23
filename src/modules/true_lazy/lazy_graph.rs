use core::mem::transmute;

use super::ty::{Graphable, Type};

#[derive(Debug, Default)]
pub struct LazyGraph {
    operations: Vec<(Type, *mut dyn Fn(&'static ()))>,
}

impl Drop for LazyGraph {
    fn drop(&mut self) {
        for (ty, operation) in self.operations.iter_mut() {
            match ty {
                Type::F32 => {
                    let operation =
                        unsafe { transmute::<_, &mut *mut dyn Fn(&mut [f32])>(operation) };
                    unsafe { drop(Box::from_raw(*operation)) }
                }
                Type::I32 => {
                    let operation =
                        unsafe { transmute::<_, &mut *mut dyn Fn(&mut [i32])>(operation) };
                    unsafe { drop(Box::from_raw(*operation)) }
                }
            }
        }
    }
}

impl LazyGraph {
    pub fn add_operation<T: Graphable>(&mut self, operation: impl Fn(&mut [T])) {
        let operation = Box::leak(Box::new(operation));
        self.operations
            .push((T::TYPE, operation as *mut dyn Fn(&mut [T]) as *mut _))
    }

    pub fn call_lazily(&mut self) {
        for (ty, operation) in self.operations.iter_mut() {
            match ty {
                Type::F32 => {
                    let operation =
                        unsafe { transmute::<_, &mut *mut dyn Fn(&mut [f32])>(operation) };
                    let mut out = vec![0f32; 100];
                    unsafe {
                        (**operation)(&mut out);
                    }
                }
                Type::I32 => {
                    let operation =
                        unsafe { transmute::<_, &mut *mut dyn Fn(&mut [i32])>(operation) };

                    let mut out = vec![0i32; 100];
                    unsafe {
                        (**operation)(&mut out);
                    }
                    println!("out: {out:?}")
                }
            }
        }
    }
}
