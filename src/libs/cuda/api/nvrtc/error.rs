use crate::Error;


pub enum NvrtcErrorKind {
    Unknown
}

pub type NvrtcResult<T> = std::result::Result<T, NvrtcErrorKind>;

impl NvrtcErrorKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            _ => "Unknown"
        }
    }
}

impl From<u32> for NvrtcErrorKind {
    fn from(value: u32) -> Self {
        println!("nvrtc value: {value}");
        match value {
            _ => NvrtcErrorKind::Unknown
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

impl From<NvrtcErrorKind> for Error {
    fn from(error: NvrtcErrorKind) -> Self {
        Error { error: Box::new(error) }
    }
}

impl std::error::Error for NvrtcErrorKind {}