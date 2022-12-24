
#[cfg(not(feature = "no-std"))]
mod std_err {
    pub type Error = Box<dyn std::error::Error + Send + Sync>;

    pub trait ErrorKind {
        fn kind<E: std::error::Error + PartialEq + 'static>(&self) -> Option<&E>;
    }
    
    impl ErrorKind for Error {
        fn kind<E: std::error::Error + PartialEq + 'static>(&self) -> Option<&E> {
            self.downcast_ref::<E>()
        }
    }

    impl std::error::Error for crate::DeviceError {}
}

#[cfg(not(feature="no-std"))]
pub use std_err::Error;


/* 
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

impl<T: std::error::Error + Send + 'static + Sync> From<T> for Error {
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
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.error)?;
        Ok(())
    }
}
*/

#[cfg(not(feature = "no-std"))]
pub type Result<T> = core::result::Result<T, self::std_err::Error>;

#[cfg(feature = "no-std")]
#[derive(Debug)]
pub struct Error {}

#[cfg(feature = "no-std")]
pub type Result<T> = core::result::Result<T, Error>;

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DeviceError {
    ConstructError,
    CPUtoCUDA,
    GraphOptimization, // probably a programming error
    MissingAddress,
}

impl DeviceError {
    pub fn as_str(&self) -> &'static str {
        match self {
            DeviceError::ConstructError => {
                "Only a non-drop buffer can be converted to a CPU+OpenCL buffer."
            }
            DeviceError::CPUtoCUDA => "Only a CPU Buffer can be converted to a CUDA Buffer",
            DeviceError::GraphOptimization => "This graph can't be optimized.",
            DeviceError::MissingAddress => "An address was not supplied for a Network device.",
        }
    }
}

impl core::fmt::Debug for DeviceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl core::fmt::Display for DeviceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}


