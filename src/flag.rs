/// Descripes the type of a [`Buffer`]
#[derive(Debug, Clone, Copy, Eq, Default)]
pub enum AllocFlag {
    #[default]
    None,
    Cache,
    Wrapper,
}


impl PartialEq for AllocFlag {
    fn eq(&self, other: &Self) -> bool {
        core::mem::discriminant(self) == core::mem::discriminant(other)
    }
}
