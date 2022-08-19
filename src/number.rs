use core::{
    cmp::Ordering,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use std::iter::Sum;

#[cfg(feature="half")]
use half::{f16, bf16};

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

macro_rules! impl_number {
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

impl_number! {
    f32, f64, i8, i16, i32, i64, i128,
    isize, u8, u16, u32, u64, u128, usize
}

#[cfg(feature="half")]
macro_rules! impl_number_half {
    ($($t:ident),*) => {
        $(
            impl Number for $t {
                #[inline]
                fn from_usize(value: usize) -> Self {
                    $t::from(value as u8)
                }

                #[inline]
                fn from_u64(value: u64) -> Self {
                    $t::from(value as u8)
                }

                #[inline]
                fn as_usize(&self) -> usize {
                    self.to_f64() as usize
                }

                #[inline]
                fn as_f64(&self) -> f64 {
                    self.to_f64()
                }

                #[inline]
                fn zero() -> Self {
                    $t::from_bits(0)
                }

                #[inline]
                fn one() -> Self {
                    $t::from_bits(15360)
                }

                #[inline]
                fn two() -> Self {
                    $t::from_bits(16384)
                }
            }
        )*

    };
}

#[cfg(feature="half")]
impl_number_half!(f16, bf16);

pub trait Float: Neg<Output = Self> + Number {
    fn negate(&self) -> Self;
    fn squared(lhs: Self) -> Self;
    fn exp(&self) -> Self;
    fn powf(&self, rhs: Self) -> Self;
    fn powi(&self, rhs: i32) -> Self;
    fn comp(lhs: Self, rhs: Self) -> Option<Ordering>;
    //fn from_usize(value: usize) -> Self;
    fn tanh(&self) -> Self;
    fn sin(&self) -> Self;
    fn as_generic(value: f64) -> Self;
    fn sqrt(&self) -> Self;
    fn ln(&self) -> Self;
    fn abs(&self) -> Self;
}

macro_rules! impl_float {
    ($($t:ident),*) => {
        $(
            impl Float for $t {
                #[inline]
                fn negate(&self) -> $t {
                    use core::ops::Neg;
                    self.neg()
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
                fn comp(lhs: $t, rhs: $t) -> Option<Ordering> {
                    lhs.partial_cmp(&rhs)
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

impl_float!(f32, f64);

#[cfg(feature="half")]
macro_rules! impl_float_half {
    ($($t:ident),*) => {
        $(
            impl Float for $t {
                fn negate(&self) -> Self {
                    self.neg()
                }
            
                fn squared(lhs: Self) -> Self {
                    lhs*lhs
                }
            
                fn exp(&self) -> Self {
                    $t::from_f32(self.to_f32().exp())
                }
            
                fn powf(&self, rhs: Self) -> Self {
                    $t::from_f32(self.to_f32().powf(rhs.to_f32()))
                }
            
                fn powi(&self, rhs: i32) -> Self {
                    $t::from_f32(self.to_f32().powi(rhs))
                }
            
                fn comp(lhs: Self, rhs: Self) -> Option<Ordering> {
                    lhs.partial_cmp(&rhs)
                }
            
                fn tanh(&self) -> Self {
                    $t::from_f32(self.to_f32().tanh())
                }
            
                fn sin(&self) -> Self {
                    $t::from_f32(self.to_f32().sin())
                }
            
                fn as_generic(value: f64) -> Self {
                    $t::from_f64(value)
                }
            
                fn sqrt(&self) -> Self {
                    $t::from_f32(self.to_f32().sqrt())
                }
            
                fn ln(&self) -> Self {
                    $t::from_f32(self.to_f32().ln())
                }
            
                fn abs(&self) -> Self {
                    $t::from_f32(self.to_f32().abs())
                }
            }
        )*
    };
}

#[cfg(feature="half")]
impl_float_half!(f16, bf16);