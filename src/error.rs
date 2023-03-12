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

#[cfg(not(feature = "no-std"))]
pub use std_err::*;

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
    GraphOptimization, // probably a bug
    MissingAddress,
    WGPUDeviceReturn,
}

impl DeviceError {
    pub fn as_str(&self) -> &'static str {
        match self {
            DeviceError::ConstructError => {
                "Only a non-drop buffer can be converted to a CPU+OpenCL buffer."
            }
            DeviceError::CPUtoCUDA => "Only a CPU Buffer can be converted to a CUDA Buffer",
            DeviceError::GraphOptimization => {
                "This graph can't be optimized. This indicates a bug in custos."
            }
            DeviceError::MissingAddress => "An address was not supplied for a Network device.",
            DeviceError::WGPUDeviceReturn => "Cannot create WGPU device instance.",
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
