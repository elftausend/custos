//! Describes the type of allocation.

/// Descripes the type of allocation.
#[derive(Debug, Clone, Copy, Eq, PartialOrd, Ord, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AllocFlag {
    #[default]
    /// Typically used for temporary buffers. These buffers are not cached and are deallocated if they go out of scope.
    None,
    /// Wraps around another pointer. Such buffers are not deallocated when they go out of scope.
    Wrapper,
    /// If a Buffer / allocation only contains a single number.
    Num,
    /// Similiar to `None`, but the resulting [`Buffer`](crate::Buffer) is borrowed and not owned.
    BorrowedCache,
}

impl PartialEq for AllocFlag {
    fn eq(&self, other: &Self) -> bool {
        core::mem::discriminant(self) == core::mem::discriminant(other)
    }
}
