pub enum WebGlError {
    MissingWindow,
    MissingDocument,
    CanvasCreation,
    DataCreation,
    BufferCreation,
    ContextCreation,
    MissingShader,
    FrameBufferCreation,
    DatatypeArgumentMismatch,
    ArgumentCountMismatch,
    OutputBufferSizeMismatch,
}

impl core::fmt::Display for WebGlError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            WebGlError::MissingWindow => "Cannot get js Window object",
            WebGlError::MissingDocument => "Cannot get document of Window object",
            WebGlError::CanvasCreation => "Cannot create canvas element",
            WebGlError::DataCreation => "Cannot create webgl data",
            WebGlError::BufferCreation => "Cannot create webgl buffer",
            WebGlError::ContextCreation => "Cannot create webgl context",
            WebGlError::MissingShader => "Missing compute shader in provided source",
            WebGlError::FrameBufferCreation => "Cannot create frame buffer",
            WebGlError::DatatypeArgumentMismatch => {
                "Mismatch of shader input datatype and provided argument"
            }
            WebGlError::ArgumentCountMismatch => {
                "Mismatch of global variable amount and provided argument array length"
            }
            WebGlError::OutputBufferSizeMismatch => "All output buffers require the same size",
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
