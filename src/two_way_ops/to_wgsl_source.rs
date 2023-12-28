use crate::{Combiner, ToCLSource};
pub trait ToWgslSource {
    fn to_wgsl_source(&self) -> String;
}

impl<T: Combiner + ToCLSource> ToWgslSource for T {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        self.to_cl_source()
    }
}

// TODO: --------- these functions are never called, self.to_cl_source() never calls these

#[cfg(feature = "half")]
impl ToWgslSource for half::f16 {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        format!("f16({:?})", self)
    }
}

// impl ToWgslSource for f32 {
//     #[inline]
//     fn to_wgsl_source(&self) -> String {
//         format!("f32({:?})", self)
//     }
// }

// #[cfg(not(target_os = "macos"))]
// impl ToWgslSource for f64 {
//     #[inline]
//     fn to_wgsl_source(&self) -> String {
//         println!("f64");
//         format!("f64({:?})", self)
//     }
// }

// impl ToWgslSource for i32 {
//     #[inline]
//     fn to_wgsl_source(&self) -> String {
//         format!("i32({:?})", self)
//     }
// }

// impl ToWgslSource for u32 {
//     #[inline]
//     fn to_wgsl_source(&self) -> String {
//         format!("u32({:?})", self)
//     }
// }

impl ToWgslSource for &'static str {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        self.to_string()
    }
}

impl ToWgslSource for String {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        self.to_string()
    }
}

pub trait MayToWgslSource: ToWgslSource {}
impl<T: ToWgslSource> MayToWgslSource for T {}
