use crate::{prelude::Float, Combiner, Eval};

#[cfg(feature = "std")]
use super::ToCLSource;

pub struct Identity<C> {
    pub comb: C,
}

impl<C> Combiner for Identity<C> {}

impl<T: Float, C: Eval<T>> Eval<T> for Identity<C> {
    #[inline]
    fn eval(&self) -> T {
        self.comb.eval()
    }
}

#[cfg(feature = "std")]
impl<C: ToCLSource> ToCLSource for Identity<C> {
    #[inline]
    fn to_cl_source(&self) -> String {
        self.comb.to_cl_source()
    }
}

pub struct Exp<C> {
    pub comb: C,
}

impl<C> Combiner for Exp<C> {}

impl<T: Float, C: Eval<T>> Eval<T> for Exp<C> {
    #[inline]
    fn eval(&self) -> T {
        Float::exp(&self.comb.eval())
        // self.comb.eval().exp()
    }
}

#[cfg(feature = "std")]
impl<C: ToCLSource> ToCLSource for Exp<C> {
    #[inline]
    fn to_cl_source(&self) -> String {
        format!("exp({})", self.comb.to_cl_source())
    }
}

pub struct Sin<C> {
    pub comb: C,
}

impl<C> Combiner for Sin<C> {}

impl<T: Float, C: Eval<T>> Eval<T> for Sin<C> {
    #[inline]
    fn eval(&self) -> T {
        Float::sin(&self.comb.eval())
        // self.comb.eval().sin()
    }
}

#[cfg(feature = "std")]
impl<C: ToCLSource> ToCLSource for Sin<C> {
    #[inline]
    fn to_cl_source(&self) -> String {
        format!("sin({})", self.comb.to_cl_source())
    }
}

pub struct Cos<C> {
    pub comb: C,
}

impl<C> Combiner for Cos<C> {}

impl<T: Float, C: Eval<T>> Eval<T> for Cos<C> {
    #[inline]
    fn eval(&self) -> T {
        Float::cos(&self.comb.eval())
        // self.comb.eval().cos()
    }
}

#[cfg(feature = "std")]
impl<C: ToCLSource> ToCLSource for Cos<C> {
    #[inline]
    fn to_cl_source(&self) -> String {
        format!("cos({})", self.comb.to_cl_source())
    }
}

pub struct Tan<C> {
    pub comb: C,
}

impl<C> Combiner for Tan<C> {}

impl<T: Float, C: Eval<T>> Eval<T> for Tan<C> {
    #[inline]
    fn eval(&self) -> T {
        Float::tan(&self.comb.eval())
        // self.comb.eval().tan()
    }
}

#[cfg(feature = "std")]
impl<C: ToCLSource> ToCLSource for Tan<C> {
    #[inline]
    fn to_cl_source(&self) -> String {
        format!("tan({})", self.comb.to_cl_source())
    }
}

pub struct Tanh<C> {
    pub comb: C,
}

impl<C> Combiner for Tanh<C> {}

impl<T: Float, C: Eval<T>> Eval<T> for Tanh<C> {
    #[inline]
    fn eval(&self) -> T {
        Float::tanh(&self.comb.eval())
    }
}

#[cfg(feature = "std")]
impl<C: ToCLSource> ToCLSource for Tanh<C> {
    #[inline]
    fn to_cl_source(&self) -> String {
        format!("tanh({})", self.comb.to_cl_source())
    }
}

pub struct Neg<C> {
    pub comb: C,
}

impl<C> Combiner for Neg<C> {}

impl<T: core::ops::Neg<Output = T>, C: Eval<T>> Eval<T> for Neg<C> {
    #[inline]
    fn eval(&self) -> T {
        self.comb.eval().neg()
    }
}

#[cfg(feature = "std")]
impl<C: ToCLSource> ToCLSource for Neg<C> {
    #[inline]
    fn to_cl_source(&self) -> String {
        format!("-({})", self.comb.to_cl_source())
    }
}

pub struct Ln<C> {
    pub comb: C,
}

impl<C> Combiner for Ln<C> {}

impl<T: Float, C: Eval<T>> Eval<T> for Ln<C> {
    #[inline]
    fn eval(&self) -> T {
        Float::ln(&self.comb.eval())
    }
}

#[cfg(feature = "std")]
impl<C: ToCLSource> ToCLSource for Ln<C> {
    #[inline]
    fn to_cl_source(&self) -> String {
        format!("log({})", self.comb.to_cl_source())
    }
}
