pub enum NvrtcErrorKind {
    OutOfMemory,
    ProgramCreationFailure,
    InvalidInput,
    InvalidProgram,
    InvalidOption,
    Compilation,
    BuiltinOperationFailure,
    NoNameExpressionsAfterCompilation,
    NoLoweredNamesBeforeCompilation,
    NameExpressionNotValid,
    InternalError,
    Unknown,
}

pub type NvrtcResult<T> = std::result::Result<T, NvrtcErrorKind>;

impl NvrtcErrorKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            NvrtcErrorKind::OutOfMemory => "NVRTC_ERROR_OUT_OF_MEMORY",
            NvrtcErrorKind::ProgramCreationFailure => "NVRTC_ERROR_PROGRAM_CREATION_FAILURE",
            NvrtcErrorKind::InvalidInput => "NVRTC_ERROR_INVALID_INPUT",
            NvrtcErrorKind::InvalidProgram => "NVRTC_ERROR_INVALID_PROGRAM",
            NvrtcErrorKind::InvalidOption => "NVRTC_ERROR_INVALID_OPTION ",
            NvrtcErrorKind::Compilation => "NVRTC_ERROR_COMPILATION",
            NvrtcErrorKind::BuiltinOperationFailure => "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE",
            NvrtcErrorKind::NoNameExpressionsAfterCompilation => {
                "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION"
            }
            NvrtcErrorKind::NoLoweredNamesBeforeCompilation => {
                "NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION"
            }
            NvrtcErrorKind::NameExpressionNotValid => "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID",
            NvrtcErrorKind::InternalError => "NVRTC_ERROR_INTERNAL_ERROR",
            NvrtcErrorKind::Unknown => "Unknown NVRTC error",
        }
    }
}

impl From<u32> for NvrtcErrorKind {
    fn from(value: u32) -> Self {
        match value {
            1 => NvrtcErrorKind::OutOfMemory,
            2 => NvrtcErrorKind::ProgramCreationFailure,
            3 => NvrtcErrorKind::InvalidInput,
            4 => NvrtcErrorKind::InvalidProgram,
            5 => NvrtcErrorKind::InvalidOption,
            6 => NvrtcErrorKind::Compilation,
            7 => NvrtcErrorKind::BuiltinOperationFailure,
            8 => NvrtcErrorKind::NoNameExpressionsAfterCompilation,
            9 => NvrtcErrorKind::NoLoweredNamesBeforeCompilation,
            10 => NvrtcErrorKind::NameExpressionNotValid,
            11 => NvrtcErrorKind::InternalError,
            _ => NvrtcErrorKind::Unknown,
        }
    }
}

impl core::fmt::Debug for NvrtcErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())?;
        Ok(())
    }
}

impl core::fmt::Display for NvrtcErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())?;
        Ok(())
    }
}

impl std::error::Error for NvrtcErrorKind {}
