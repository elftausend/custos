use std::cell::RefCell;

pub mod opencl;
pub mod cpu;

thread_local! {
    pub static COUNT: RefCell<usize> = RefCell::new(0);
}

pub fn set_count(count: usize) {
    COUNT.with(|c| *c.borrow_mut() = count);
}

pub fn get_count() -> usize {
    COUNT.with(|c| *c.borrow())
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Node {
    idx: usize,
    out_dims: (usize, usize),
}

impl Node {
    pub fn new(out_dims: (usize, usize)) -> Node {
        COUNT.with(|count| {
            let node = Node {
                idx: *count.borrow(),
                out_dims,
                
            };
            *count.borrow_mut() += 1;
            node
        })
    }
}