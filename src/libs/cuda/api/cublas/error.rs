use crate::Error;


pub enum CublasErrorKind {
    Unknown
}

pub type CublasResult<T> = std::result::Result<T, CublasErrorKind>;

impl CublasErrorKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            _ => "Unknown"
        }
    }
}

impl From<u32> for CublasErrorKind {
    fn from(value: u32) -> Self {
        println!("value: {value}");
        match value {
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