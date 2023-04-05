use core::cell::Cell;
use std::thread_local;

thread_local! {
    pub(crate) static COUNT: Cell<usize> = Cell::new(0);
}

/// Sets current cache identifier / index.
/// This function is usually called after an iteration in a loop -> [Count](crate::Count) or [range](crate::range)
/// # Safety
/// Manually setting the count may yield multiple `Buffer` pointing two the same data.
#[inline]
pub unsafe fn set_count(count: usize) {
    COUNT.with(|c| c.set(count));
}

/// Returns current cache identifier / index
#[inline]
pub fn get_count() -> usize {
    COUNT.with(|c| c.get())
}

#[inline]
/// Increases the cache identifier / index by 1.
pub fn bump_count() {
    COUNT.with(|c| {
        let count = c.get();
        c.set(count + 1);
    })
}

pub trait IdAble {
    fn id(self) -> Option<Ident>;
}

impl IdAble for () {
    #[inline]
    fn id(self) -> Option<Ident> {
        None
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
/// An `Ident` is used to identify a cached pointer.
pub struct Ident {
    /// The index of the `Ident`.
    pub idx: usize,
    /// The amount of elements a corresponding [`Buffer`](crate::Buffer) has.
    pub len: usize,
}

impl Ident {
    /// Returns a new `Ident` with the current cache identifier / index.
    #[inline]
    pub fn new(len: usize) -> Ident {
        Ident {
            idx: get_count(),
            len,
        }
    }

    /// Returns a new `Ident` with the current cache identifier / index and increases the cache identifier / index by 1.
    #[inline]
    pub fn new_bumped(len: usize) -> Ident {
        let id = Ident {
            idx: get_count(),
            len,
        };
        bump_count();
        id
    }
}

impl IdAble for Option<Ident> {
    #[inline]
    fn id(self) -> Option<Ident> {
        self
    }
}
