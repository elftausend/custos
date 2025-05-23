pub trait ToWgslSource {
    fn to_wgsl_source(&self) -> String;
}

#[cfg(feature = "half")]
impl ToWgslSource for half::f16 {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        format!("f16({self:?})")
    }
}

impl ToWgslSource for f32 {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        format!("f32({self:?})")
    }
}

// #[cfg(not(target_os = "macos"))]
// impl ToWgslSource for f64 {
//     #[inline]
//     fn to_wgsl_source(&self) -> String {
//         format!("f64({:?})", self)
//     }
// }

impl ToWgslSource for i32 {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        format!("i32({self:?})")
    }
}

impl ToWgslSource for u32 {
    #[inline]
    fn to_wgsl_source(&self) -> String {
        format!("u32({self:?})")
    }
}

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

macro_rules! wgsl_unsupported_datatypes {
    ($($t:ident),*) => {
        $(
            impl ToWgslSource for $t {
                #[inline]
                fn to_wgsl_source(&self) -> String {
                    unimplemented!("This scalar datatype ({}) is not supported by WGSL.", core::any::type_name::<$t>());
                }
            }
        )*
    };
}

wgsl_unsupported_datatypes! {
    f64, i8, i16, i64, i128,
    isize, u8, u16, u64, u128, usize
}
