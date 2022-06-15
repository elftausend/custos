use crate::Error;

pub type CudaResult<T> = std::result::Result<T, CudaErrorKind>;

pub enum CudaErrorKind {
    InvalidAllocSize,
    InvalidDeviceIdx,
    Unknown
}

impl CudaErrorKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            CudaErrorKind::InvalidDeviceIdx => "Invalid device idx, specific CUDA device not found",
            _ => "Unknown"
        }
    }
}

impl From<u32> for CudaErrorKind {
    fn from(value: u32) -> Self {
        println!("cuda value: {value}");
        match value {
            _ => CudaErrorKind::Unknown
        }
    }
}

impl core::fmt::Debug for CudaErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())?;
        Ok(())
    }
}

impl core::fmt::Display for CudaErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())?;
        Ok(())
    }
}

impl From<CudaErrorKind> for Error {
    fn from(error: CudaErrorKind) -> Self {
        Error { error: Box::new(error) }
    }
}

impl std::error::Error for CudaErrorKind {}