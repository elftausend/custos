use core::fmt::Display;

use super::{Combiner, Eval};

#[derive(Debug, Clone, Copy)]
pub struct Resolve<T> {
    pub val: T,
    pub marker: &'static str,
}

pub trait ToMarker<T, R> {
    fn to_marker(self) -> R;
}

impl<T: Default> ToMarker<T, Resolve<T>> for &'static str {
    #[inline]
    fn to_marker(self) -> Resolve<T> {
        Resolve::with_marker(self)
    }
}

impl<T: Default> ToMarker<T, (Resolve<T>, Resolve<T>)> for (&'static str, &'static str) {
    #[inline]
    fn to_marker(self) -> (Resolve<T>, Resolve<T>) {
        (Resolve::with_marker(self.0), Resolve::with_marker(self.1))
    }
}

pub trait ToVal<T> {
    fn to_val(self) -> Resolve<T>;
}

impl<T> ToVal<T> for T {
    #[inline]
    fn to_val(self) -> Resolve<T> {
        Resolve::new(self)
    }
}

impl<T: Default> Default for Resolve<T> {
    #[inline]
    fn default() -> Self {
        Self {
            val: T::default(),
            marker: "x",
        }
    }
}

impl<T> Resolve<T> {
    #[inline]
    pub fn new(val: T) -> Self {
        Resolve { val, marker: "x" }
    }

    #[inline]
    pub fn with_marker(marker: &'static str) -> Self
    where
        T: Default,
    {
        Resolve {
            val: T::default(),
            marker,
        }
    }
}

impl<T> Eval<T> for Resolve<T> {
    #[inline]
    fn eval(self) -> T {
        self.val
    }
}

impl<T: Display> ToString for Resolve<T> {
    #[inline]
    fn to_string(&self) -> String {
        format!("{}", self.marker)
    }
}

impl<T> Combiner for Resolve<T> {}
