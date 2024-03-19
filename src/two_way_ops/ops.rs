mod cmps;
mod unary;

use crate::{prelude::Float, ToWgslSource};

#[cfg(feature = "std")]
use crate::ToCLSource;

use super::{Combiner, Eval};
pub use cmps::*;
pub use unary::*;

// TODO: maybe use a macro to generate these

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

#[cfg(feature = "std")]
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

#[cfg(feature = "std")]
impl<C: ToWgslSource, R: ToWgslSource> ToWgslSource for Mul<C, R> {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        format!(
            "({} * {})",
            self.comb.to_wgsl_source(),
            self.rhs.to_wgsl_source()
        )
    }
}

impl<C: Eval<T>, R: Eval<T>, T: core::ops::Mul<Output = T>> Eval<T> for Mul<C, R> {
    #[inline]
    fn eval(&self) -> T {
        self.comb.eval() * self.rhs.eval()
    }
}

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

#[cfg(feature = "std")]
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

#[cfg(feature = "std")]
impl<C: ToWgslSource, R: ToWgslSource> ToWgslSource for Add<C, R> {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        format!(
            "({} + {})",
            self.comb.to_wgsl_source(),
            self.rhs.to_wgsl_source()
        )
    }
}

impl<C: Eval<T>, R: Eval<T>, T: core::ops::Add<Output = T>> Eval<T> for Add<C, R> {
    #[inline]
    fn eval(&self) -> T {
        self.comb.eval() + self.rhs.eval()
    }
}

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

#[cfg(feature = "std")]
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

#[cfg(feature = "std")]
impl<C: ToWgslSource, R: ToWgslSource> ToWgslSource for Sub<C, R> {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        format!(
            "({} - {})",
            self.comb.to_wgsl_source(),
            self.rhs.to_wgsl_source()
        )
    }
}

impl<C: Eval<T>, R: Eval<T>, T: core::ops::Sub<Output = T>> Eval<T> for Sub<C, R> {
    #[inline]
    fn eval(&self) -> T {
        self.comb.eval() - self.rhs.eval()
    }
}

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

#[cfg(feature = "std")]
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

#[cfg(feature = "std")]
impl<C: ToWgslSource, R: ToWgslSource> ToWgslSource for Div<C, R> {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        format!(
            "({} / {})",
            self.comb.to_wgsl_source(),
            self.rhs.to_wgsl_source()
        )
    }
}

impl<C: Eval<T>, R: Eval<T>, T: core::ops::Div<Output = T>> Eval<T> for Div<C, R> {
    #[inline]
    fn eval(&self) -> T {
        self.comb.eval() / self.rhs.eval()
    }
}

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

#[cfg(feature = "std")]
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

#[cfg(feature = "std")]
impl<C: ToWgslSource, R: ToWgslSource> ToWgslSource for Pow<C, R> {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        format!(
            "pow({}, {})",
            self.comb.to_wgsl_source(),
            self.rhs.to_wgsl_source()
        )
    }
}

impl<C: Eval<T>, R: Eval<T>, T: Float> Eval<T> for Pow<C, R> {
    #[inline]
    fn eval(&self) -> T {
        self.comb.eval().powf(self.rhs.eval())
    }
}

pub struct Min<C, R> {
    pub comb: C,
    pub rhs: R,
}

impl<C, R> Combiner for Min<C, R> {}

#[cfg(feature = "std")]
impl<C: ToCLSource, R: ToCLSource> ToCLSource for Min<C, R> {
    #[inline]
    fn to_cl_source(&self) -> String {
        format!(
            "min({}, {})",
            self.comb.to_cl_source(),
            self.rhs.to_cl_source()
        )
    }
}

#[cfg(feature = "std")]
impl<C: ToWgslSource, R: ToWgslSource> ToWgslSource for Min<C, R> {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        format!(
            "min({}, {})",
            self.comb.to_wgsl_source(),
            self.rhs.to_wgsl_source()
        )
    }
}

impl<C: Eval<T>, R: Eval<T>, T: Float> Eval<T> for Min<C, R> {
    #[inline]
    fn eval(&self) -> T {
        self.comb.eval().min(self.rhs.eval())
    }
}

pub struct Max<C, R> {
    pub comb: C,
    pub rhs: R,
}

impl<C, R> Combiner for Max<C, R> {}

#[cfg(feature = "std")]
impl<C: ToCLSource, R: ToCLSource> ToCLSource for Max<C, R> {
    #[inline]
    fn to_cl_source(&self) -> String {
        format!(
            "max({}, {})",
            self.comb.to_cl_source(),
            self.rhs.to_cl_source()
        )
    }
}

#[cfg(feature = "std")]
impl<C: ToWgslSource, R: ToWgslSource> ToWgslSource for Max<C, R> {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        format!(
            "max({}, {})",
            self.comb.to_wgsl_source(),
            self.rhs.to_wgsl_source()
        )
    }
}

impl<C: Eval<T>, R: Eval<T>, T: Float> Eval<T> for Max<C, R> {
    #[inline]
    fn eval(&self) -> T {
        self.comb.eval().max(self.rhs.eval())
    }
}
