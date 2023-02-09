/// Descripes the type of a [`Buffer`]
#[derive(Debug, Clone, Copy, Eq)]
pub enum AllocFlag {
    None,
    Cache,
    Wrapper,
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
