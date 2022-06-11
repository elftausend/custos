use crate::Error;

pub type CudaResult<T> = std::result::Result<T, CudaErrorKind>;

pub enum CudaErrorKind {
    Test,
    InvalidAllocSize
}

impl CudaErrorKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            _ => "Test"
        }
    }
}

impl From<u32> for CudaErrorKind {
    fn from(value: u32) -> Self {
        println!("value: {value}");
        match value {
            _ => CudaErrorKind::Test
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