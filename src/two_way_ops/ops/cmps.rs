use crate::{prelude::Number, Eval, Combiner};

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

impl<C: ToString, R: ToString> ToString for GEq<C, R> {
    fn to_string(&self) -> String {
        format!("({} >= {})", self.comb.to_string(), self.rhs.to_string())
    }
}


impl<C: Eval<T>, R: Eval<T>, T: Number> Eval<T> for GEq<C, R> {
    #[inline]
    fn eval(self) -> T {
        T::from_usize(self.comb.eval().ge(&self.rhs.eval()) as usize)
    }
}

impl<C, R> Combiner for GEq<C, R> {}

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

impl<C: ToString, R: ToString> ToString for LEq<C, R> {
    fn to_string(&self) -> String {
        format!("({} <= {})", self.comb.to_string(), self.rhs.to_string())
    }
}


impl<C: Eval<T>, R: Eval<T>, T: Number> Eval<T> for LEq<C, R> {
    #[inline]
    fn eval(self) -> T {
        T::from_usize(self.comb.eval().le(&self.rhs.eval()) as usize)
    }
}

impl<C, R> Combiner for LEq<C, R> {}

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

impl<C: ToString, R: ToString> ToString for Eq<C, R> {
    fn to_string(&self) -> String {
        format!("({} <= {})", self.comb.to_string(), self.rhs.to_string())
    }
}


impl<C: Eval<T>, R: Eval<T>, T: Number> Eval<T> for Eq<C, R> {
    #[inline]
    fn eval(self) -> T {
        T::from_usize(self.comb.eval().le(&self.rhs.eval()) as usize)
    }
}

impl<C, R> Combiner for Eq<C, R> {}
