use core::{cmp::Ordering, ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign}};
use rand::distributions::uniform::SampleUniform;

macro_rules! number_apply {
    ($t:ident) => {
        impl Number for $t {
            fn from_usize(value: usize) -> $t {
                value as $t
            }
            fn from_u64(value: u64) -> $t {
                value as $t
            }
            fn as_usize(&self) -> usize {
                *self as usize
            }
            fn as_f64(&self) -> f64 {
                *self as f64
            }
            fn zero() -> $t {
                0 as $t
            }
            fn one() -> $t {
                1 as $t
            }
            fn two() -> $t {
                2 as $t
            }
            /* 
            fn one() -> $t {
                1 as $t
            }
            */

        }
    };
}

number_apply!(f32);
number_apply!(f64);
number_apply!(i8);
number_apply!(i16);
number_apply!(i32);
number_apply!(i64);
number_apply!(i128);
number_apply!(isize);
number_apply!(u8);
number_apply!(u16);
number_apply!(u32);
number_apply!(u64);
number_apply!(u128);
number_apply!(usize);


pub trait Number: Sized+Default+Clone+Copy+
                    Add<Self, Output = Self>+
                    Sub<Self, Output = Self>+
                    Div<Self, Output = Self>+
                    Mul<Self, Output = Self>+

                    Add<&'static Self, Output = Self>+
                    Sub<&'static Self, Output = Self>+
                    Div<&'static Self, Output = Self>+
                    Mul<&'static Self, Output = Self>+
                    AddAssign<Self> + SubAssign<Self>+
                    MulAssign<Self> + DivAssign<Self>+
                    SampleUniform + PartialOrd+
                    PartialEq +
                    'static + core::fmt::Debug + core::fmt::Display + {

        fn from_usize(value: usize) -> Self;
        fn from_u64(value: u64) -> Self;
        fn as_usize(&self) -> usize;
        fn as_f64(&self) -> f64;
        fn zero() -> Self;
        fn one() -> Self;
        fn two() -> Self;
}

macro_rules! float_apply {
    ($t:ident) => {
        impl Float for $t {
            fn negate(&self) -> $t {
                use core::ops::Neg;
                self.neg()
                
            }
            fn squared(lhs: $t) -> $t {
                lhs*lhs
            }
            fn exp(&self) -> $t {
                $t::exp(*self)
            } 
            fn powf(&self, rhs: $t) -> $t {
                $t::powf(*self, rhs)
            }
            fn powi(&self, rhs: i32) -> $t {
                $t::powi(*self, rhs)
            }
            fn comp(lhs: $t, rhs: $t) -> Option<Ordering> {
                lhs.partial_cmp(&rhs)
            }
            fn tanh(&self) -> $t {
                $t::tanh(*self)
            }
            fn sin(&self) -> $t {
                $t::sin(*self)
            }
            fn as_generic(value: f64) -> $t {
                value as $t
            }
            fn sqrt(&self) -> $t {
                $t::sqrt(*self)
            }
            fn ln(&self) -> $t {
                $t::ln(*self)
            }
            fn abs(&self) -> $t {
                $t::abs(*self)
            }            
        }
    };
}

float_apply!(f32);
float_apply!(f64);

#[cfg(feature="opencl")]
pub trait Float:
            Neg + Number + TGenericOCL
    {
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
#[cfg(not(feature="opencl"))]
pub trait Float:
            Neg + Number
    {
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
