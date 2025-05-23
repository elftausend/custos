//! Describes the type of allocation.

/// Descripes the type of allocation.
#[derive(Debug, Clone, Copy, Eq, PartialOrd, Ord, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AllocFlag {
    #[default]
    /// Typically used for temporary buffers. These buffers are deallocated when they go out of scope.
    None,
    /// Typically used for cached buffers. These buffers are deallocated when they go out of scope.
    Cached,
    /// Wraps around another pointer. Such buffers are not deallocated when they go out of scope.
    Wrapper,
}

impl PartialEq for AllocFlag {
    fn eq(&self, other: &Self) -> bool {
        core::mem::discriminant(self) == core::mem::discriminant(other)
    }
}

impl AllocFlag {
    #[inline]
    pub fn continue_deallocation(&self) -> bool {
        matches!(self, AllocFlag::None | AllocFlag::Cached)
    }
}
