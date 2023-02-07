mod cmps;
mod unary;

use crate::prelude::Float;

use super::{Combiner, Eval};
pub use cmps::*;
pub use unary::*;

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

impl<C: ToString, R: ToString> ToString for Mul<C, R> {
    #[inline]
    fn to_string(&self) -> String {
        format!("({} * {})", self.comb.to_string(), self.rhs.to_string())
    }
}

impl<C: Eval<T>, R: Eval<T>, T: std::ops::Mul<Output = T>> Eval<T> for Mul<C, R> {
    #[inline]
    fn eval(self) -> T {
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

impl<C: ToString, R: ToString> ToString for Add<C, R> {
    #[inline]
    fn to_string(&self) -> String {
        format!("({} + {})", self.comb.to_string(), self.rhs.to_string())
    }
}

impl<C: Eval<T>, R: Eval<T>, T: std::ops::Add<Output = T>> Eval<T> for Add<C, R> {
    #[inline]
    fn eval(self) -> T {
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

impl<C: ToString, R: ToString> ToString for Sub<C, R> {
    #[inline]
    fn to_string(&self) -> String {
        format!("({} - {})", self.comb.to_string(), self.rhs.to_string())
    }
}

impl<C: Eval<T>, R: Eval<T>, T: std::ops::Sub<Output = T>> Eval<T> for Sub<C, R> {
    #[inline]
    fn eval(self) -> T {
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

impl<C: ToString, R: ToString> ToString for Div<C, R> {
    #[inline]
    fn to_string(&self) -> String {
        format!("({} / {})", self.comb.to_string(), self.rhs.to_string())
    }
}

impl<C: Eval<T>, R: Eval<T>, T: std::ops::Div<Output = T>> Eval<T> for Div<C, R> {
    #[inline]
    fn eval(self) -> T {
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

impl<C: ToString, R: ToString> ToString for Pow<C, R> {
    #[inline]
    fn to_string(&self) -> String {
        format!("pow({}, {})", self.comb.to_string(), self.rhs.to_string())
    }
}

impl<C: Eval<T>, R: Eval<T>, T: Float> Eval<T> for Pow<C, R> {
    #[inline]
    fn eval(self) -> T {
        self.comb.eval().powf(self.rhs.eval())
    }
}
