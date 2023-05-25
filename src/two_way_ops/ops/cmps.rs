use crate::{prelude::Number, Combiner, Eval};

#[cfg(not(feature = "no-std"))]
use super::ToCLSource;

#[derive(Clone)]
pub struct GEq<C, R> {
    pub comb: C,
    pub rhs: R,
}

impl<C, R> GEq<C, R> {
    #[inline]
    pub fn new(comb: C, rhs: R) -> GEq<C, R> {
        GEq { comb, rhs }
    }
}

#[cfg(not(feature = "no-std"))]
impl<C: ToCLSource, R: ToCLSource> ToCLSource for GEq<C, R> {
    #[inline]
    fn to_cl_source(&self) -> String {
        format!(
            "({} >= {})",
            self.comb.to_cl_source(),
            self.rhs.to_cl_source()
        )
    }
}

impl<C: Eval<T>, R: Eval<T>, T: Number> Eval<T> for GEq<C, R> {
    #[inline]
    fn eval(self, input: T) -> T {
        T::from_usize(self.comb.eval(input).ge(&self.rhs.eval(input)) as usize)
    }
}

impl<C, R> Combiner for GEq<C, R> {}

#[derive(Clone)]
pub struct LEq<C, R> {
    pub comb: C,
    pub rhs: R,
}

impl<C, R> LEq<C, R> {
    #[inline]
    pub fn new(comb: C, rhs: R) -> LEq<C, R> {
        LEq { comb, rhs }
    }
}

#[cfg(not(feature = "no-std"))]
impl<C: ToCLSource, R: ToCLSource> ToCLSource for LEq<C, R> {
    #[inline]
    fn to_cl_source(&self) -> String {
        format!(
            "({} <= {})",
            self.comb.to_cl_source(),
            self.rhs.to_cl_source()
        )
    }
}

impl<C: Eval<T>, R: Eval<T>, T: Number> Eval<T> for LEq<C, R> {
    #[inline]
    fn eval(self, input: T) -> T {
        T::from_usize(self.comb.eval(input).le(&self.rhs.eval(input)) as usize)
    }
}

impl<C, R> Combiner for LEq<C, R> {}

#[derive(Clone)]
pub struct Eq<C, R> {
    pub comb: C,
    pub rhs: R,
}

impl<C, R> Eq<C, R> {
    #[inline]
    pub fn new(comb: C, rhs: R) -> Eq<C, R> {
        Eq { comb, rhs }
    }
}

#[cfg(not(feature = "no-std"))]
impl<C: ToCLSource, R: ToCLSource> ToCLSource for Eq<C, R> {
    #[inline]
    fn to_cl_source(&self) -> String {
        format!(
            "({} == {})",
            self.comb.to_cl_source(),
            self.rhs.to_cl_source()
        )
    }
}

impl<C: Eval<T>, R: Eval<T>, T: Number> Eval<T> for Eq<C, R> {
    #[inline]
    fn eval(self, input: T) -> T {
        T::from_usize(self.comb.eval(input).le(&self.rhs.eval(input)) as usize)
    }
}

impl<C, R> Combiner for Eq<C, R> {}
