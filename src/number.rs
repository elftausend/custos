//! Contains traits for generic math.
#![allow(missing_docs)]

use core::{
    cmp::Ordering,
    fmt::Display,
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign},
};

/// A trait that returns the default / zero of a value.
pub trait Zero {
    /// Returns zero or the default.
    /// # Example
    /// ```
    /// use custos::number::Zero;
    ///
    /// assert_eq!(f32::zero(), 0.);
    ///
    /// ```
    #[inline]
    fn zero() -> Self
    where
        Self: Default,
    {
        Self::default()
    }
}

impl<T: Default> Zero for T {}

/// A trait that returns 1 for a number.
pub trait One {
    /// Returns one..
    /// # Example
    /// ```
    /// use custos::number::One;
    ///
    /// fn generic_one<T: One>() -> T {
    ///     T::one()
    /// }
    ///
    /// assert_eq!(generic_one::<f32>(), 1.);
    ///
    /// ```
    fn one() -> Self;
}

/// A trait that returns 2 for a number.
pub trait Two {
    /// Returns two.
    /// # Example
    /// ```
    /// use custos::number::Two;
    ///
    /// fn generic_two<T: Two>() -> T {
    ///     T::two()
    /// }
    ///
    /// assert_eq!(generic_two::<f32>(), 2.);
    ///
    /// ```
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

#[cfg(feature = "half")]
impl One for half::f16 {
    #[inline]
    fn one() -> Self {
        half::f16::ONE
    }
}

#[cfg(feature = "half")]
impl One for half::bf16 {
    #[inline]
    fn one() -> Self {
        half::bf16::ONE
    }
}

#[cfg(feature = "half")]
impl Two for half::f16 {
    #[inline]
    fn two() -> Self {
        half::f16::ONE
    }
}

#[cfg(feature = "half")]
impl Two for half::bf16 {
    #[inline]
    fn two() -> Self {
        half::bf16::ONE
    }
}

/// Numeric is a trait that is implemented for all numeric types.
pub trait Numeric:
    Sized + Default + Copy + PartialOrd + PartialEq + core::fmt::Debug + core::fmt::Display + 'static
{
}

impl Numeric for bool {}
impl Numeric for f32 {}
impl Numeric for f64 {}
impl Numeric for i8 {}
impl Numeric for i16 {}
impl Numeric for i32 {}
impl Numeric for i64 {}
impl Numeric for i128 {}
impl Numeric for isize {}
impl Numeric for u8 {}
impl Numeric for u16 {}
impl Numeric for u32 {}
impl Numeric for u64 {}
impl Numeric for u128 {}
impl Numeric for usize {}

#[cfg(feature = "half")]
impl Numeric for half::f16 {}

#[cfg(feature = "half")]
impl Numeric for half::bf16 {}

/// Implementors of `Number` require some basic math operations.
/// # Example
/// ```
/// use custos::number::Number;
///
/// fn generic_add<T: Number>(a: T, b: T) -> T {
///     a + b
/// }
///
/// assert_eq!(generic_add(1., 2.), 3.);
/// ```
pub trait Number:
    Numeric
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Div<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Rem<Self, Output = Self>
    + One
    + Two
    + Zero
    + for<'a> Rem<&'a Self, Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Div<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + RemAssign<Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + DivAssign<Self>
    + Sum<Self>
    + Display
{
    fn from_usize(value: usize) -> Self;
    fn from_u64(value: u64) -> Self;
    fn as_usize(&self) -> usize;
    fn as_f64(&self) -> f64;

    #[inline]
    fn max(self, rhs: Self) -> Self {
        if self > rhs {
            self
        } else {
            rhs
        }
    }

    #[inline]
    fn min(self, rhs: Self) -> Self {
        if self < rhs {
            self
        } else {
            rhs
        }
    }
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
    fn from_f64(value: f64) -> Self;
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

#[cfg(feature = "std")]
macro_rules! float_apply {
    ($($t:ident),*) => {
        $(
            impl Float for $t {
                #[inline]
                fn from_f64(value: f64) -> $t {
                    value as $t
                }

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

#[cfg(feature = "std")]
float_apply!(f32, f64);

#[cfg(not(feature = "std"))]
#[cfg(feature = "no-std")]
impl Float for f32 {
    #[inline]
    fn from_f64(value: f64) -> f32 {
        value as f32
    }

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
        (*self > 0.) as usize as f32 * *self - (*self < 0.) as usize as f32 * *self
    }

    #[inline]
    fn cos(&self) -> Self {
        libm::cosf(*self)
    }

    #[inline]
    fn tan(&self) -> Self {
        libm::tanf(*self)
    }

    #[inline]
    fn log(&self, base: Self) -> Self {
        libm::log10f(*self) / libm::log10f(base)
    }
}

#[cfg(not(feature = "std"))]
#[cfg(feature = "no-std")]
impl Float for f64 {
    #[inline]
    fn from_f64(value: f64) -> f64 {
        value
    }

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
        (*self > 0.) as usize as f64 * *self - (*self < 0.) as usize as f64 * *self
    }

    #[inline]
    fn cos(&self) -> Self {
        libm::cos(*self)
    }

    #[inline]
    fn tan(&self) -> Self {
        libm::tan(*self)
    }

    #[inline]
    fn log(&self, base: Self) -> Self {
        libm::log10(*self) / libm::log10(base)
    }
}

#[cfg(feature = "half")]
impl Number for half::f16 {
    #[inline]
    fn from_usize(value: usize) -> Self {
        half::f16::from_f32(value as f32)
    }

    #[inline]
    fn from_u64(value: u64) -> Self {
        half::f16::from_f32(value as f32)
    }

    #[inline]
    fn as_usize(&self) -> usize {
        self.to_f32() as usize
    }

    #[inline]
    fn as_f64(&self) -> f64 {
        self.to_f64()
    }

    #[inline]
    fn max(self, rhs: Self) -> Self {
        self.max(rhs)
    }
}

#[cfg(feature = "half")]
impl Number for half::bf16 {
    #[inline]
    fn from_usize(value: usize) -> Self {
        half::bf16::from_f32(value as f32)
    }

    #[inline]
    fn from_u64(value: u64) -> Self {
        half::bf16::from_f32(value as f32)
    }

    #[inline]
    fn as_usize(&self) -> usize {
        self.to_f32() as usize
    }

    #[inline]
    fn as_f64(&self) -> f64 {
        self.to_f64()
    }

    #[inline]
    fn max(self, rhs: Self) -> Self {
        self.max(rhs)
    }
}

#[cfg(feature = "half")]
impl Float for half::f16 {
    #[inline]
    fn exp(&self) -> Self {
        Self::from_f32(self.to_f32().exp())
    }

    #[inline]
    fn powf(&self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32().powf(rhs.to_f32()))
    }

    #[inline]
    fn powi(&self, rhs: i32) -> Self {
        Self::from_f32(self.to_f32().powi(rhs))
    }

    #[inline]
    fn tanh(&self) -> Self {
        Self::from_f32(self.to_f32().tanh())
    }

    #[inline]
    fn sin(&self) -> Self {
        Self::from_f32(self.to_f32().sin())
    }

    #[inline]
    fn cos(&self) -> Self {
        Self::from_f32(self.to_f32().cos())
    }

    #[inline]
    fn tan(&self) -> Self {
        Self::from_f32(self.to_f32().cos())
    }

    #[inline]
    fn as_generic(value: f64) -> Self {
        Self::from_f64(value)
    }

    #[inline]
    fn sqrt(&self) -> Self {
        Self::from_f32(self.to_f32().sqrt())
    }

    #[inline]
    fn log(&self, base: Self) -> Self {
        Self::from_f32(self.to_f32().log(base.to_f32()))
    }

    #[inline]
    fn ln(&self) -> Self {
        Self::from_f32(self.to_f32().ln())
    }

    #[inline]
    fn abs(&self) -> Self {
        Self::from_f32(self.to_f32().abs())
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        Self::from_f64(value)
    }
}

#[cfg(feature = "half")]
impl Float for half::bf16 {
    #[inline]
    fn exp(&self) -> Self {
        Self::from_f32(self.to_f32().exp())
    }

    #[inline]
    fn powf(&self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32().powf(rhs.to_f32()))
    }

    #[inline]
    fn powi(&self, rhs: i32) -> Self {
        Self::from_f32(self.to_f32().powi(rhs))
    }

    #[inline]
    fn tanh(&self) -> Self {
        Self::from_f32(self.to_f32().tanh())
    }

    #[inline]
    fn sin(&self) -> Self {
        Self::from_f32(self.to_f32().sin())
    }

    #[inline]
    fn cos(&self) -> Self {
        Self::from_f32(self.to_f32().cos())
    }

    #[inline]
    fn tan(&self) -> Self {
        Self::from_f32(self.to_f32().cos())
    }

    #[inline]
    fn as_generic(value: f64) -> Self {
        Self::from_f64(value)
    }

    #[inline]
    fn sqrt(&self) -> Self {
        Self::from_f32(self.to_f32().sqrt())
    }

    #[inline]
    fn log(&self, base: Self) -> Self {
        Self::from_f32(self.to_f32().log(base.to_f32()))
    }

    #[inline]
    fn ln(&self) -> Self {
        Self::from_f32(self.to_f32().ln())
    }

    #[inline]
    fn abs(&self) -> Self {
        Self::from_f32(self.to_f32().abs())
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        Self::from_f64(value)
    }
}
