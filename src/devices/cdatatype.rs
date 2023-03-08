
/// enables easy generic kernel creation
pub trait CDatatype: 'static {
    // TODO: this would make more sense as an associated constant
    fn as_c_type_str() -> &'static str;
}

impl CDatatype for bool {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "bool"
    }
}

#[cfg(any(not(target_os = "macos"), not(feature = "opencl")))]
impl CDatatype for f64 {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "double"
    }
}

impl CDatatype for f32 {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "float"
    }
}

impl CDatatype for i32 {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "int"
    }
}

impl CDatatype for u32 {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "uint"
    }
}

impl CDatatype for i8 {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "char"
    }
}

impl CDatatype for u8 {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "uchar"
    }
}

impl CDatatype for i16 {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "short"
    }
}
impl CDatatype for u16 {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "ushort"
    }
}

impl CDatatype for i64 {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "long"
    }
}

impl CDatatype for u64 {
    #[inline]
    fn as_c_type_str() -> &'static str {
        "ulong"
    }
}
