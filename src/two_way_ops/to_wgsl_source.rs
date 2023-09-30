use crate::{Combiner, ToCLSource};

#[cfg(not(feature = "no-std"))]
pub trait ToWgslSource {
    fn to_wgsl_source(&self) -> String;
}

#[cfg(not(feature = "no-std"))]
impl<T: Combiner + ToCLSource> ToWgslSource for T {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        self.to_cl_source()
    }
}

// TODO: --------- these functions are never called, self.to_cl_source() never calls these

#[cfg(feature = "half")]
#[cfg(not(feature = "no-std"))]
impl ToWgslSource for half::f16 {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        format!("f16({:?})", self)
    }
}

#[cfg(not(feature = "no-std"))]
impl ToWgslSource for f32 {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        format!("f32({:?})", self)
    }
}

#[cfg(not(target_os = "macos"))]
#[cfg(not(feature = "no-std"))]
impl ToWgslSource for f64 {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        println!("f64");
        format!("f64({:?})", self)
    }
}

#[cfg(not(feature = "no-std"))]
impl ToWgslSource for i32 {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        format!("i32({:?})", self)
    }
}

#[cfg(not(feature = "no-std"))]
impl ToWgslSource for u32 {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        format!("u32({:?})", self)
    }
}

#[cfg(not(feature = "no-std"))]
impl ToWgslSource for &'static str {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        self.to_string()
    }
}

#[cfg(not(feature = "no-std"))]
impl ToWgslSource for String {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        self.to_string()
    }
}

#[cfg(not(feature = "no-std"))]
pub trait MayToWgslSource: ToWgslSource {}
#[cfg(not(feature = "no-std"))]
impl<T: ToWgslSource> MayToWgslSource for T {}
