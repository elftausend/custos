use crate::{prelude::Float, Combiner, Eval};

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

impl<C: ToString> ToString for Sin<C> {
    #[inline]
    fn to_string(&self) -> String {
        format!("sin({})", self.comb.to_string())
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

impl<C: ToString> ToString for Cos<C> {
    #[inline]
    fn to_string(&self) -> String {
        format!("cos({})", self.comb.to_string())
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

impl<C: ToString> ToString for Tan<C> {
    #[inline]
    fn to_string(&self) -> String {
        format!("tan({})", self.comb.to_string())
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

impl<C: ToString> ToString for Neg<C> {
    #[inline]
    fn to_string(&self) -> String {
        format!("-({})", self.comb.to_string())
    }
}