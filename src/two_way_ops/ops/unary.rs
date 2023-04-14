use crate::{prelude::Float, Combiner, Eval};

#[cfg(not(feature = "no-std"))]
use super::ToCLSource;

pub struct Exp<C> {
    pub comb: C,
}

impl<C> Combiner for Exp<C> {}

impl<T: Float, C: Eval<T>> Eval<T> for Exp<C> {
    #[inline]
    fn eval(self) -> T {
        self.comb.eval().exp()
    }
}

#[cfg(not(feature = "no-std"))]
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
    fn eval(self) -> T {
        self.comb.eval().sin()
    }
}

#[cfg(not(feature = "no-std"))]
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
    fn eval(self) -> T {
        self.comb.eval().cos()
    }
}

#[cfg(not(feature = "no-std"))]
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
    fn eval(self) -> T {
        self.comb.eval().tan()
    }
}

#[cfg(not(feature = "no-std"))]
impl<C: ToCLSource> ToCLSource for Tan<C> {
    #[inline]
    fn to_cl_source(&self) -> String {
        format!("tan({})", self.comb.to_cl_source())
    }
}

pub struct Neg<C> {
    pub comb: C,
}

impl<C> Combiner for Neg<C> {}

impl<T: core::ops::Neg<Output = T>, C: Eval<T>> Eval<T> for Neg<C> {
    #[inline]
    fn eval(self) -> T {
        self.comb.eval().neg()
    }
}

#[cfg(not(feature = "no-std"))]
impl<C: ToCLSource> ToCLSource for Neg<C> {
    #[inline]
    fn to_cl_source(&self) -> String {
        format!("-({})", self.comb.to_cl_source())
    }
}
