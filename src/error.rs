pub struct Error {
    pub error: Box<dyn std::error::Error + Send>,
}

impl<E: std::error::Error + PartialEq + 'static> PartialEq<E> for Error {
    fn eq(&self, other: &E) -> bool {
        let e = self.error.downcast_ref::<E>();
        if let Some(e) = e {
            return e == other;
        }
        false
    }
}

impl From<Error> for Box<dyn std::error::Error> {
    fn from(e: Error) -> Self {
        e.error
    }
}

impl Error {
    pub fn kind<E: std::error::Error + PartialEq + 'static>(&self) -> Option<&E> {
        self.error.downcast_ref::<E>()
    }
}

impl<T: std::error::Error + Send + 'static> From<T> for Error {
    fn from(error: T) -> Self {
        Error {
            error: Box::new(error),
        }
    }
}

impl core::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.error)?;
        Ok(())
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.error)?;
        Ok(())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DeviceError {
    ConstructError,
    CPUtoCUDA,
    GraphOptimization, // probably a programming error
}

impl DeviceError {
    pub fn as_str(&self) -> &'static str {
        match self {
            DeviceError::ConstructError => {
                "Only a non-drop buffer can be converted to a CPU+OpenCL buffer."
            }
            DeviceError::CPUtoCUDA => "Only a CPU Buffer can be converted to a CUDA Buffer",
            DeviceError::GraphOptimization => "This graph can't be optimized.",
        }
    }
}

impl core::fmt::Debug for DeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl core::fmt::Display for DeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for DeviceError {}
