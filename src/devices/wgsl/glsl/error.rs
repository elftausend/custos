use core::fmt::{Debug, Display};

pub enum GlslError {
    CreationError,
    CompileError(String),
}

impl Display for GlslError {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GlslError::CreationError => write!(f, "Could not create shader."),
            GlslError::CompileError(info) => write!(f, "Error during compilation: {info}"),
        }
    }
}

impl Debug for GlslError {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        Display::fmt(self, f)
    }
}
