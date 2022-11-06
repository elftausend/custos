/// Descripes the type of a [`Buffer`]
#[derive(Debug, Clone, Copy)]
pub enum BufFlag {
    None,
    Cache,
    Wrapper,
}

impl Default for BufFlag {
    fn default() -> Self {
        BufFlag::None
    }
}

impl PartialEq for BufFlag {
    fn eq(&self, other: &Self) -> bool {
        core::mem::discriminant(self) == core::mem::discriminant(other)
    }
}
