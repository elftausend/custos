/// Evaluates a combined (via [`Combiner`]) math operations chain to a valid OpenCL C (and possibly CUDA) source string.
pub trait ToCLSource {
    /// Evaluates a combined (via [`Combiner`]) math operations chain to a valid OpenCL C (and possibly CUDA) source string.
    fn to_cl_source(&self) -> String;
}

impl<N: crate::number::Numeric> ToCLSource for N {
    #[inline]
    fn to_cl_source(&self) -> String {
        format!("{:?}", self)
    }
}

impl ToCLSource for &'static str {
    #[inline]
    fn to_cl_source(&self) -> String {
        self.to_string()
    }
}

impl ToCLSource for String {
    #[inline]
    fn to_cl_source(&self) -> String {
        self.to_string()
    }
}

/// If the `no-std` feature is disabled, this trait is implemented for all types that implement [`ToCLSource`].
/// In this case, `no-std` is disabled.
pub trait MayToCLSource: ToCLSource {}
impl<T: ToCLSource> MayToCLSource for T {}
