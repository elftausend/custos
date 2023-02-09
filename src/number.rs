use core::{
    cmp::Ordering,
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

pub trait Number:
    Sized
    + Default
    + Copy
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Div<Self, Output = Self>
    + Mul<Self, Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + DivAssign<Self>
    + PartialOrd
    + PartialEq
    + core::fmt::Debug
    + core::fmt::Display
    + Sum<Self>
{
    fn from_usize(value: usize) -> Self;
    fn from_u64(value: u64) -> Self;
    fn as_usize(&self) -> usize;
    fn as_f64(&self) -> f64;
    fn zero() -> Self;
    fn one() -> Self;
    fn two() -> Self;
}

macro_rules! number_apply {
    ($($t:ident),*) => {
        $(
            impl Number for $t {
                #[inline]
                fn from_usize(value: usize) -> $t {
                    value as $t
                }

                #[inline]
                fn from_u64(value: u64) -> $t {
                    value as $t
                }

                #[inline]
                fn as_usize(&self) -> usize {
                    *self as usize
                }

                #[inline]
                fn as_f64(&self) -> f64 {
                    *self as f64
                }

                #[inline]
                fn zero() -> $t {
                    0 as $t
                }

                #[inline]
                fn one() -> $t {
                    1 as $t
                }

                #[inline]
                fn two() -> $t {
                    2 as $t
                }
            }
        )*

    };
}

number_apply! {
    f32, f64, i8, i16, i32, i64, i128,
    isize, u8, u16, u32, u64, u128, usize
}

pub trait Float: Neg<Output = Self> + Number {
    #[inline]
    fn squared(lhs: Self) -> Self {
        lhs * lhs
    }
    fn exp(&self) -> Self;
    fn powf(&self, rhs: Self) -> Self;
    fn powi(&self, rhs: i32) -> Self;

    #[inline]
    fn cmp(lhs: Self, rhs: Self) -> Option<Ordering> {
        lhs.partial_cmp(&rhs)
    }
    //fn from_usize(value: usize) -> Self;
    fn tanh(&self) -> Self;
    fn sin(&self) -> Self;
    fn as_generic(value: f64) -> Self;
    fn sqrt(&self) -> Self;
    fn ln(&self) -> Self;
    fn abs(&self) -> Self;
}

#[cfg(not(feature = "no-std"))]
macro_rules! float_apply {
    ($($t:ident),*) => {
        $(
            impl Float for $t {

                #[inline]
                fn squared(lhs: $t) -> $t {
                    lhs*lhs
                }
                #[inline]
                fn exp(&self) -> $t {
                    $t::exp(*self)
                }
                #[inline]
                fn powf(&self, rhs: $t) -> $t {
                    $t::powf(*self, rhs)
                }
                #[inline]
                fn powi(&self, rhs: i32) -> $t {
                    $t::powi(*self, rhs)
                }

                #[inline]
                fn tanh(&self) -> $t {
                    $t::tanh(*self)
                }
                #[inline]
                fn sin(&self) -> $t {
                    $t::sin(*self)
                }
                #[inline]
                fn as_generic(value: f64) -> $t {
                    value as $t
                }
                #[inline]
                fn sqrt(&self) -> $t {
                    $t::sqrt(*self)
                }
                #[inline]
                fn ln(&self) -> $t {
                    $t::ln(*self)
                }
                #[inline]
                fn abs(&self) -> $t {
                    $t::abs(*self)
                }
            }
        )*
    };
}

#[cfg(not(feature = "no-std"))]
float_apply!(f32, f64);

#[cfg(feature = "no-std")]
impl Float for f32 {
    #[inline]
    fn exp(&self) -> Self {
        libm::expf(*self)
    }

    #[inline]
    fn powf(&self, rhs: Self) -> Self {
        libm::powf(*self, rhs)
    }

    #[inline]
    fn powi(&self, rhs: i32) -> Self {
        libm::powf(*self, rhs as f32)
    }

    #[inline]
    fn tanh(&self) -> Self {
        libm::tanhf(*self)
    }

    #[inline]
    fn sin(&self) -> Self {
        libm::sinf(*self)
    }

    #[inline]
    fn as_generic(value: f64) -> Self {
        value as f32
    }

    #[inline]
    fn sqrt(&self) -> Self {
        libm::sqrtf(*self)
    }

    #[inline]
    fn ln(&self) -> Self {
        libm::logf(*self)
    }

    #[inline]
    fn abs(&self) -> Self {
        self * ((*self > 0.) as u32 - (*self < 0.) as u32) as f32
    }
}

#[cfg(feature = "no-std")]
impl Float for f64 {
    #[inline]
    fn exp(&self) -> Self {
        libm::exp(*self)
    }

    #[inline]
    fn powf(&self, rhs: Self) -> Self {
        libm::pow(*self, rhs)
    }

    #[inline]
    fn powi(&self, rhs: i32) -> Self {
        libm::pow(*self, rhs as f64)
    }

    #[inline]
    fn tanh(&self) -> Self {
        libm::tanh(*self)
    }

    #[inline]
    fn sin(&self) -> Self {
        libm::sin(*self)
    }

    #[inline]
    fn as_generic(value: f64) -> Self {
        value
    }

    #[inline]
    fn sqrt(&self) -> Self {
        libm::sqrt(*self)
    }

    #[inline]
    fn ln(&self) -> Self {
        libm::log(*self)
    }

    #[inline]
    fn abs(&self) -> Self {
        self * ((*self > 0.) as u64 - (*self < 0.) as u64) as f64
    }
}
