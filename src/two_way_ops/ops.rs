use crate::prelude::Float;

use super::{Combiner, Eval};

pub struct Sin<C> {
    pub comb: C,
}

impl<C> Combiner for Sin<C> {}

impl<T: Float, C: Eval<T>> Eval<T> for Sin<C> {
    fn eval(self) -> T {
        self.comb.eval().sin()
    }
}

impl<C: ToString> ToString for Sin<C> {
    fn to_string(&self) -> String {
        format!("sin({})", self.comb.to_string())
    }
}

pub struct Cos<C> {
    pub comb: C,
}

impl<C> Combiner for Cos<C> {}

impl<T: Float, C: Eval<T>> Eval<T> for Cos<C> {
    fn eval(self) -> T {
        self.comb.eval().cos()
    }
}

impl<C: ToString> ToString for Cos<C> {
    fn to_string(&self) -> String {
        format!("cos({})", self.comb.to_string())
    }
}

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
