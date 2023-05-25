mod cmps;
mod unary;

use crate::prelude::Float;

#[cfg(not(feature = "no-std"))]
use crate::ToCLSource;

use super::{Combiner, Eval};
pub use cmps::*;
pub use unary::*;

#[derive(Clone)]
pub struct Mul<C, R> {
    comb: C,
    rhs: R,
}

impl<C, R> Mul<C, R> {
    #[inline]
    pub fn new(comb: C, rhs: R) -> Mul<C, R> {
        Mul { comb, rhs }
    }
}

impl<C, R> Combiner for Mul<C, R> {}

#[cfg(not(feature = "no-std"))]
impl<C: ToCLSource, R: ToCLSource> ToCLSource for Mul<C, R> {
    #[inline]
    fn to_cl_source(&self) -> String {
        format!(
            "({} * {})",
            self.comb.to_cl_source(),
            self.rhs.to_cl_source()
        )
    }
}

impl<C: Eval<T>, R: Eval<T>, T: core::ops::Mul<Output = T> + Copy> Eval<T> for Mul<C, R> {
    #[inline]
    fn eval(self, input: T) -> T {
        self.comb.eval(input) * self.rhs.eval(input)
    }
}

#[derive(Clone)]
pub struct Add<C, R> {
    comb: C,
    rhs: R,
}

impl<C, R> Add<C, R> {
    #[inline]
    pub fn new(comb: C, rhs: R) -> Add<C, R> {
        Add { comb, rhs }
    }
}

impl<C, R> Combiner for Add<C, R> {}

#[cfg(not(feature = "no-std"))]
impl<C: ToCLSource, R: ToCLSource> ToCLSource for Add<C, R> {
    #[inline]
    fn to_cl_source(&self) -> String {
        format!(
            "({} + {})",
            self.comb.to_cl_source(),
            self.rhs.to_cl_source()
        )
    }
}

impl<C: Eval<T>, R: Eval<T>, T: core::ops::Add<Output = T> + Copy> Eval<T> for Add<C, R> {
    #[inline]
    fn eval(self, input: T) -> T {
        self.comb.eval(input) + self.rhs.eval(input)
    }
}

#[derive(Clone)]
pub struct Sub<C, R> {
    comb: C,
    rhs: R,
}

impl<C, R> Sub<C, R> {
    #[inline]
    pub fn new(comb: C, rhs: R) -> Sub<C, R> {
        Sub { comb, rhs }
    }
}

impl<C, R> Combiner for Sub<C, R> {}

#[cfg(not(feature = "no-std"))]
impl<C: ToCLSource, R: ToCLSource> ToCLSource for Sub<C, R> {
    #[inline]
    fn to_cl_source(&self) -> String {
        format!(
            "({} - {})",
            self.comb.to_cl_source(),
            self.rhs.to_cl_source()
        )
    }
}

impl<C: Eval<T>, R: Eval<T>, T: core::ops::Sub<Output = T> + Copy> Eval<T> for Sub<C, R> {
    #[inline]
    fn eval(self, input: T) -> T {
        self.comb.eval(input) - self.rhs.eval(input)
    }
}

#[derive(Clone)]
pub struct Div<C, R> {
    comb: C,
    rhs: R,
}

impl<C, R> Div<C, R> {
    #[inline]
    pub fn new(comb: C, rhs: R) -> Div<C, R> {
        Div { comb, rhs }
    }
}

impl<C, R> Combiner for Div<C, R> {}

#[cfg(not(feature = "no-std"))]
impl<C: ToCLSource, R: ToCLSource> ToCLSource for Div<C, R> {
    #[inline]
    fn to_cl_source(&self) -> String {
        format!(
            "({} / {})",
            self.comb.to_cl_source(),
            self.rhs.to_cl_source()
        )
    }
}

impl<C: Eval<T>, R: Eval<T>, T: core::ops::Div<Output = T> + Copy> Eval<T> for Div<C, R> {
    #[inline]
    fn eval(self, input: T) -> T {
        self.comb.eval(input) / self.rhs.eval(input)
    }
}

#[derive(Clone)]
pub struct Pow<C, R> {
    comb: C,
    rhs: R,
}

impl<C, R> Pow<C, R> {
    #[inline]
    pub fn new(comb: C, rhs: R) -> Pow<C, R> {
        Pow { comb, rhs }
    }
}

impl<C, R> Combiner for Pow<C, R> {}

#[cfg(not(feature = "no-std"))]
impl<C: ToCLSource, R: ToCLSource> ToCLSource for Pow<C, R> {
    #[inline]
    fn to_cl_source(&self) -> String {
        format!(
            "pow({}, {})",
            self.comb.to_cl_source(),
            self.rhs.to_cl_source()
        )
    }
}

impl<C: Eval<T>, R: Eval<T>, T: Float> Eval<T> for Pow<C, R> {
    #[inline]
    fn eval(self, input: T) -> T {
        self.comb.eval(input).powf(self.rhs.eval(input))
    }
}
