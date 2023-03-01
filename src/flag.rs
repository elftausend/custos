use core::default;

/// Descripes the type of a [`Buffer`]
#[derive(Debug, Clone, Copy, Eq, PartialOrd, Ord, Default)]
pub enum AllocFlag {
    #[default]
    None,
    Cache,
    Wrapper,
    Num,
    BorrowedCache,
}

impl PartialEq for AllocFlag {
    fn eq(&self, other: &Self) -> bool {
        core::mem::discriminant(self) == core::mem::discriminant(other)
    }
}
