/*use std::rc::{Weak, Rc};

thread_local! {
    pub static CPU_CACHE: Vec<(Rc<*mut usize>, usize)> = vec![];
}

pub struct CacheBuffer<T> {
    ptr: Weak<*mut T>,
    len: usize,
}

#[test]
fn test_cpu_cache() {
    let x = Rc::new(10.);
    let weak = Rc::downgrade(&x);
    let upgrade = weak.upgrade().unwrap();
    let x = Rc::as_ptr(&x) as *mut f64;
}*/
