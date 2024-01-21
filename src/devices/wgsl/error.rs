use core::fmt::Display;

use naga::{WithSpan, valid::ValidationError};


#[derive(Debug)]
pub enum TranslateError {
    Validate(WithSpan<ValidationError>),
    Frontend(naga::front::wgsl::ParseError),
    #[cfg(feature = "spv")]
    BackendSpv(naga::back::spv::Error),
    #[cfg(feature = "glsl")]
    BackendGlsl(naga::back::glsl::Error),
}

impl Display for TranslateError {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TranslateError::Validate(validation_err) => validation_err.fmt(f),
            TranslateError::Frontend(frontend_err) => frontend_err.fmt(f),
            #[cfg(feature = "spv")]
            TranslateError::BackendSpv(backend_err) => backend_err.fmt(f),
            #[cfg(feature = "glsl")]
            TranslateError::BackendGlsl(backend_err) => backend_err.fmt(f),
        }
    }
}

impl std::error::Error for TranslateError {}
