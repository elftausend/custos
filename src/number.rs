use core::{
    cmp::Ordering,
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign},
};

pub trait Zero {
    #[inline]
    fn zero() -> Self
    where
        Self: Default,
    {
        Self::default()
    }
}

impl<T: Default> Zero for T {}

pub trait One {
    fn one() -> Self;
}

pub trait Two {
    fn two() -> Self;
}

macro_rules! typical_number_impl {
    ($($t:ident),*) => {
        $(
            impl One for $t {
                #[inline]
                fn one() -> $t {
                    1 as $t
                }
            }

            impl Two for $t {
                #[inline]
                fn two() -> $t {
                    2 as $t
                }
            }
        )*

    };
}

typical_number_impl! {
    f32, f64, i8, i16, i32, i64, i128,
    isize, u8, u16, u32, u64, u128, usize
}

pub trait Number:
    Sized
    + Default
    + Copy
    + One
    + Two
    + Zero
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Div<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Rem<Self, Output = Self>
    + for<'a> Rem<&'a Self, Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Div<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + DivAssign<Self>
    + RemAssign<Self>
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
    fn cos(&self) -> Self;
    fn tan(&self) -> Self;
    fn as_generic(value: f64) -> Self;
    fn sqrt(&self) -> Self;
    fn log(&self, base: Self) -> Self;
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
                fn cos(&self) -> $t {
                    $t::cos(*self)
                }

                #[inline]
                fn tan(&self) -> $t {
                    $t::tan(*self)
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
                fn log(&self, base: Self) -> $t {
                    $t::log(*self, base)
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
