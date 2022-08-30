pub type CudaResult<T> = std::result::Result<T, CudaErrorKind>;

pub enum CudaErrorKind {
    InvalidAllocSize,
    InvalidDeviceIdx,
    InvalidValue,
    OutOfMemory,
    NotInitialized,
    Deinitialized,
    ProfilerDisaled,
    ProfilerNotInitialized,
    ProfilerAlreadyStarted,
    ProfilerAlreadyStopped,
    NoDevice,
    InvalidDevice,
    InvalidImage,
    InvalidContext,
    ContextAlreadyCurrent,
    MapFailed,
    UnmapFailed,
    ArrayIsMapped,
    AlreadyMapped,
    NoBinaryForGPU,
    AlreadyAcquired,
    NotMapped,
    NotMappedAsArray,
    NotMappedAsPointer,
    ECCUncorrectable,
    UnsupportedLlimit,
    ContextAlreadyInUse,
    PeerAccessUnsupported,
    InvalidPTX,
    InvalidGraphicsContext,
    NVLINKUncorrectable,
    JITCompilerNotFound,
    InvalidSource,
    FileNotFound,
    SharedObjectSymbolNotFound,
    SharedObjectInitFailed,
    OperatingSystem,
    InvalidHandle,
    IllegalState,
    NotFound,
    NotReady,
    IllegalAddress,
    LaunchOutOfResources,
    LaunchTimeout,
    LaunchIncompatibleTexturing,
    Unknown,
}

impl CudaErrorKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            CudaErrorKind::InvalidDeviceIdx => "Invalid device idx, specific CUDA device not found",
            _ => "Unknown",
        }
    }
}

impl From<u32> for CudaErrorKind {
    fn from(value: u32) -> Self {
        println!("cuda value: {value}");
        CudaErrorKind::Unknown
        /*match value {
            _ => CudaErrorKind::Unknown,
        }*/
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

impl std::error::Error for CudaErrorKind {}
