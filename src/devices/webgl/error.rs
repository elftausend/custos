pub enum WebGlError {
    MissingWindow,
    MissingDocument,
    CanvasCreation,
    DataCreation,
    BufferCreation,
    ContextCreation,
}

impl core::fmt::Display for WebGlError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            WebGlError::MissingWindow => "Cannot get js Window object",
            WebGlError::MissingDocument => "Cannot get document of Window object",
            WebGlError::CanvasCreation => "Cannot create canvas element",
            WebGlError::DataCreation => "Cannot create webgl data",
            WebGlError::BufferCreation => "Cannot create webgl buffer",
            WebGlError::ContextCreation => "Cannot create webgl context"
        };
        writeln!(f, "{msg}")
    }
}

impl core::fmt::Debug for WebGlError {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(&self, f)
    }
}

impl std::error::Error for WebGlError {}
