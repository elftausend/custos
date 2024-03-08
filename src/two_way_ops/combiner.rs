use super::ops::{
    Abs, Add, Cos, Div, Eq, Exp, GEq, Identity, LEq, Ln, Max, Min, Mul, Neg, Pow, Sin, Sub, Tan, Tanh
};

/// A trait that allows combining math operations.
/// (Similiar to an Iterator)
pub trait Combiner: Sized {
    /// Combines two values into a new one via an addition.
    #[inline]
    fn add<R>(self, rhs: R) -> Add<Self, R> {
        Add::new(self, rhs)
    }

    /// Combines two values into a new one via an multiplication.
    #[inline]
    fn mul<R>(self, rhs: R) -> Mul<Self, R> {
        Mul::new(self, rhs)
    }

    /// Combines two values into a new one via an subtraction.
    #[inline]
    fn sub<R>(self, rhs: R) -> Sub<Self, R> {
        Sub::new(self, rhs)
    }

    /// Combines two values into a new one via an division.
    #[inline]
    fn div<R>(self, rhs: R) -> Div<Self, R> {
        Div::new(self, rhs)
    }

    /// Calculates the sine of a value.
    #[inline]
    fn sin(self) -> Sin<Self> {
        Sin { comb: self }
    }

    /// Calculates the cosine of a value.
    #[inline]
    fn cos(self) -> Cos<Self> {
        Cos { comb: self }
    }

    /// Calculates the tangent of a value.
    #[inline]
    fn tan(self) -> Tan<Self> {
        Tan { comb: self }
    }

    /// Combined two values into a new one via exponentiation.
    #[inline]
    fn pow<R>(self, rhs: R) -> Pow<Self, R> {
        Pow::new(self, rhs)
    }

    /// Checks if the left value is greater than the right value.
    #[inline]
    fn geq<R>(self, rhs: R) -> GEq<Self, R> {
        GEq { comb: self, rhs }
    }

    /// Checks if the left value is less than the right value.
    #[inline]
    fn leq<R>(self, rhs: R) -> LEq<Self, R> {
        LEq { comb: self, rhs }
    }

    /// Checks if the left value is equal to the right value.
    #[inline]
    fn eq<R>(self, rhs: R) -> Eq<Self, R> {
        Eq { comb: self, rhs }
    }

    /// Negates a value.
    #[inline]
    fn neg(self) -> Neg<Self> {
        Neg { comb: self }
    }

    /// Calculates the e^x of a value.
    #[inline]
    fn exp(self) -> Exp<Self> {
        Exp { comb: self }
    }

    #[inline]
    fn tanh(self) -> Tanh<Self> {
        Tanh { comb: self }
    }

    #[inline]
    fn identity(self) -> Identity<Self> {
        Identity { comb: self }
    }

    #[inline]
    fn min<R>(self, rhs: R) -> Min<Self, R> {
        Min { comb: self, rhs }
    }

    #[inline]
    fn max<R>(self, rhs: R) -> Max<Self, R> {
        Max { comb: self, rhs }
    }

    #[inline]
    fn ln(self) -> Ln<Self> {
        Ln { comb: self }
    }

    #[inline]
    fn abs(self) -> Abs<Self> {
        Abs { comb: self }
    }

}
