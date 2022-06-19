use crate::Error;

pub enum CublasErrorKind {
    NotInitialized,
    AllocFailed,
    InvalidValue,
    ArchMismatch,
    MappingError,
    ExecutionFailed,
    InternalError,
    NotSupported,
    LicenseError,
    Unknown
}

pub type CublasResult<T> = std::result::Result<T, CublasErrorKind>;

impl CublasErrorKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            CublasErrorKind::NotInitialized => "CUBLAS_STATUS_NOT_INITIALIZED",
            CublasErrorKind::AllocFailed => "CUBLAS_STATUS_ALLOC_FAILED",
            CublasErrorKind::InvalidValue => "CUBLAS_STATUS_INVALID_VALUE  	
            An unsupported value or parameter was passed to the function (a negative vector size, for example).
            To correct: ensure that all the parameters being passed have valid values.",
            CublasErrorKind::ArchMismatch => "CUBLAS_STATUS_ARCH_MISMATCH",
            CublasErrorKind::MappingError => "CUBLAS_STATUS_MAPPING_ERROR",
            CublasErrorKind::ExecutionFailed => "CUBLAS_STATUS_EXECUTION_FAILED",
            CublasErrorKind::InternalError => "CUBLAS_STATUS_INTERNAL_ERROR",
            CublasErrorKind::NotSupported => "CUBLAS_STATUS_NOT_SUPPORTED",
            CublasErrorKind::LicenseError => "CUBLAS_STATUS_LICENSE_ERROR",
            CublasErrorKind::Unknown => "Unknown cuBLAS error",
        }
    }
}

impl From<u32> for CublasErrorKind {
    fn from(value: u32) -> Self {
        
        match value {
            1 => CublasErrorKind::NotInitialized,
            3 => CublasErrorKind::AllocFailed,
            7 => CublasErrorKind::InvalidValue,
            8 => CublasErrorKind::ArchMismatch,
            11 => CublasErrorKind::MappingError,
            13 => CublasErrorKind::ExecutionFailed,
            14 => CublasErrorKind::InternalError,
            15 => CublasErrorKind::NotSupported,
            16 => CublasErrorKind::LicenseError,            
            _ => CublasErrorKind::Unknown
        }
    }
}

impl core::fmt::Debug for CublasErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())?;
        Ok(())
    }
}

impl core::fmt::Display for CublasErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())?;
        Ok(())
    }
}

impl From<CublasErrorKind> for Error {
    fn from(error: CublasErrorKind) -> Self {
        Error { error: Box::new(error) }
    }
}

impl std::error::Error for CublasErrorKind {}