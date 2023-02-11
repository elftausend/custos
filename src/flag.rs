/// Descripes the type of a [`Buffer`]
#[derive(Debug, Clone, Copy, Eq, PartialOrd, Ord)]
pub enum AllocFlag {
    None,
    Cache,
    Wrapper,
    Num,
}

impl Default for AllocFlag {
    fn default() -> Self {
        AllocFlag::None
    }
}

impl PartialEq for AllocFlag {
    fn eq(&self, other: &Self) -> bool {
        core::mem::discriminant(self) == core::mem::discriminant(other)
    }
}
