/// A type alias for Box<dyn core::error::Error + Send + Sync>
#[cfg(feature = "std")]
pub type Error = Box<dyn core::error::Error + Send + Sync>;
#[cfg(not(feature = "std"))]
pub type Error = DeviceError;

/// A trait for downcasting errors.
pub trait ErrorKind {
    /// Downcasts the error to the specified type.
    fn kind<E: core::error::Error + PartialEq + 'static>(&self) -> Option<&E>;
}

impl ErrorKind for Error {
    fn kind<E: core::error::Error + PartialEq + 'static>(&self) -> Option<&E> {
        #[cfg(feature = "std")]
        let err = self;

        #[cfg(not(feature = "std"))]
        let err: &dyn core::error::Error = self;
        err.downcast_ref::<E>()
    }
}

/// A type alias for `Result<T, Error>`.
pub type Result<T> = core::result::Result<T, Error>;

/// 'generic' device errors that can occur on any device.
#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DeviceError {
    /// Only a non-drop buffer can be converted to a CPU+OpenCL buffer.
    UnifiedConstructInvalidInputBuffer,
    /// Unified construction is not available for the provided modules. Add the [`Cached`](custos::Cached) module to your device.
    UnifiedConstructNotAvailable,
    /// Only a CPU Buffer can be converted to a CUDA Buffer
    CPUtoCUDA,
    /// No cache traces were given.
    MissingCacheTraces,
    /// This graph can't be optimized. This indicates a bug in custos.
    GraphOptimization, // probably a bug
    /// An address was not supplied for a Network device.
    MissingAddress,
    /// Cannot create WGPU device instance.
    WGPUDeviceReturn,
    /// The 'cpu' feature is disabled. Hence this CPU can't be created.
    CPUDeviceNotAvailable,
    /// Invalid lazy out buffer was provided in operation. Out buffer went out of scope?
    InvalidLazyBuf,
    /// Location was already used.
    LocationAlreadyInUse,
    /// Unary fusing not supported.
    UnaryFusingUnsupported,
    /// Zero length buffers are not supported
    ZeroLengthBuffer,
    /// Given generic shape length does not match with e.g. slice length
    ShapeLengthMismatch,
}

impl core::error::Error for crate::DeviceError {}

impl DeviceError {
    /// Returns a string slice containing the error message.
    pub fn as_str(&self) -> &'static str {
        match self {
            DeviceError::UnifiedConstructInvalidInputBuffer => {
                "Only a non-drop buffer can be converted to a CPU+OpenCL buffer."
            }
            DeviceError::CPUtoCUDA => "Only a CPU Buffer can be converted to a CUDA Buffer",
            DeviceError::MissingCacheTraces => "No cache traces were given",
            DeviceError::GraphOptimization => {
                "This graph can't be optimized. This indicates a bug in custos."
            }
            DeviceError::MissingAddress => "An address was not supplied for a Network device.",
            DeviceError::WGPUDeviceReturn => "Cannot create WGPU device instance.",
            DeviceError::CPUDeviceNotAvailable => {
                "The 'cpu' feature is disabled. Hence this CPU can't be created."
            }
            DeviceError::UnifiedConstructNotAvailable => {
                "Unified construction is not available for the provided modules. Add the `Cached` module to your device"
            }
            DeviceError::InvalidLazyBuf => {
                "Invalid lazy buffer was provided in operation. Did the buffer go out of scope?"
            }
            DeviceError::LocationAlreadyInUse => "Location is already in use.",
            DeviceError::UnaryFusingUnsupported => {
                "Unary fusing is not supported for this module configuration."
            }
            DeviceError::ZeroLengthBuffer => "Zero length buffers are not supported",
            DeviceError::ShapeLengthMismatch => {
                "Given generic shape length does not match with e.g. slice length"
            }
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
