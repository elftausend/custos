use core::cell::RefCell;
use std::thread_local;

thread_local! {
    pub static COUNT: RefCell<usize> = RefCell::new(0);
}

/// Sets current cache identifier / index.
/// This function is usually called after an iteration in a loop -> [Count](crate::Count) or [range](crate::range)
#[inline]
pub fn set_count(count: usize) {
    COUNT.with(|c| *c.borrow_mut() = count);
}

/// Returns current cache identifier / index
#[inline]
pub fn get_count() -> usize {
    COUNT.with(|c| *c.borrow())
}

#[inline]
/// Increases the cache identifier / index by 1.
pub fn bump_count() {
    COUNT.with(|c| *c.borrow_mut() += 1)
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
/// An `Ident` is used to identify a cached pointer.
pub struct Ident {
    pub idx: usize,
    pub len: usize,
}

impl Ident {
    pub fn new(len: usize) -> Ident {
        crate::COUNT.with(|count| Ident {
            idx: *count.borrow(),
            len,
        })
    }
}
