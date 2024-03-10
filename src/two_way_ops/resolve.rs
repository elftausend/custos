#[cfg(feature = "std")]
use crate::ToCLSource;

/// Resolves to either a mathematical expression as string or a computed value.
/// This is used to create generic kernels / operations over `OpenCL`, `CUDA` and `CPU`.
///
/// # Example
#[cfg_attr(not(feature = "no-std"), doc = "```")]
#[cfg_attr(feature = "no-std", doc = "```ignore")]
/// use custos::{Resolve, Eval, Combiner, ToCLSource};
///
/// let val = Resolve::with_val(1.5);
/// let out = val.mul(val).add(2.);
///
/// assert_eq!(out.eval(), 4.25);
///  
/// let mark = Resolve::<f32>::with_marker("x");
/// let out = mark.mul(mark).add(2.);
///
/// assert_eq!(out.to_cl_source(), "((x * x) + 2.0)");
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Resolve<T> {
    /// Acts as the seed value.
    pub val: T,
    /// Acts as the variable name for the expression.
    pub marker: &'static str,
}

/// Converts a &'static str to a [`Resolve`].
pub trait ToMarker<T, R> {
    /// Converts a &'static str to a [`Resolve`].
    /// # Example
    #[cfg_attr(not(feature = "no-std"), doc = "```")]
    #[cfg_attr(feature = "no-std", doc = "```ignore")]
    /// use custos::{Resolve, ToMarker, ToCLSource};
    ///
    /// let resolve = ToMarker::<f32, Resolve<f32>>::to_marker("x");;
    /// assert_eq!(resolve.to_cl_source(), "x");
    /// ```
    fn to_marker(self) -> R;
}

impl<T: Default> ToMarker<T, Resolve<T>> for &'static str {
    #[inline]
    fn to_marker(self) -> Resolve<T> {
        Resolve::with_marker(self)
    }
}

impl<T: Default> ToMarker<T, (Resolve<T>, Resolve<T>)> for (&'static str, &'static str) {
    #[inline]
    fn to_marker(self) -> (Resolve<T>, Resolve<T>) {
        (Resolve::with_marker(self.0), Resolve::with_marker(self.1))
    }
}

/// Converts a value to a [`Resolve`].
pub trait ToVal<T = Self> {
    /// Converts a value to a [`Resolve`].
    /// # Example
    /// ```
    /// use custos::{Resolve, ToVal, Eval};
    ///
    /// let resolve: Resolve<f32> = 1.5.to_val();
    ///
    /// assert_eq!(<Resolve<f32> as Eval<f32>>::eval(&resolve), 1.5);
    /// ```
    fn to_val(self) -> Resolve<T>;
}

impl<T> ToVal<T> for T {
    #[inline]
    fn to_val(self) -> Resolve<T> {
        Resolve::with_val(self)
    }
}

impl<T: Default> Default for Resolve<T> {
    #[inline]
    fn default() -> Self {
        Self {
            val: T::default(),
            marker: "x",
        }
    }
}

impl<T> Resolve<T> {
    /// Creates a `Resolve` with a value.
    /// # Example
    #[cfg_attr(not(feature = "no-std"), doc = "```")]
    #[cfg_attr(feature = "no-std", doc = "```ignore")]
    /// use custos::{Resolve, Eval, Combiner};
    ///
    /// let val = Resolve::with_val(1.5);
    /// let out = val.mul(val).add(2.);
    ///
    /// assert_eq!(out.eval(), 4.25);
    /// ```
    #[inline]
    pub fn with_val(val: T) -> Self {
        Resolve { val, marker: "x" }
    }

    /// Creates a `Resolve` with a marker.
    /// # Example
    #[cfg_attr(not(feature = "no-std"), doc = "```")]
    #[cfg_attr(feature = "no-std", doc = "```ignore")]
    /// use custos::{Resolve, Eval, Combiner, ToCLSource};
    ///
    /// let mark = Resolve::<f32>::with_marker("x");
    /// let out = mark.add(mark).mul(2.);
    ///
    /// assert_eq!(out.to_cl_source(), "((x + x) * 2.0)");
    /// ```
    #[inline]
    pub fn with_marker(marker: &'static str) -> Self
    where
        T: Default,
    {
        Resolve {
            val: T::default(),
            marker,
        }
    }
}

impl<T: Copy + 'static> crate::Eval<T> for Resolve<T> {
    #[inline]
    fn eval(&self) -> T {
        self.val
    }
}

#[cfg(feature = "std")]
impl<T> ToCLSource for Resolve<T> {
    #[inline]
    fn to_cl_source(&self) -> String {
        self.marker.to_string()
    }
}

impl<T> crate::Combiner for Resolve<T> {}
