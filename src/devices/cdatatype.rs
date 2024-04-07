// TODO: different types for cuda and opencl
/// enables easy generic kernel creation
pub trait CDatatype: 'static {
    const C_DTYPE_STR: &'static str;
}

impl CDatatype for bool {
    const C_DTYPE_STR: &'static str = "bool";
}

#[cfg(any(not(target_os = "macos"), not(feature = "opencl")))]
impl CDatatype for f64 {
    const C_DTYPE_STR: &'static str = "double";
}

impl CDatatype for f32 {
    const C_DTYPE_STR: &'static str = "float";
}

impl CDatatype for i32 {
    const C_DTYPE_STR: &'static str = "int";
}

impl CDatatype for u32 {
    const C_DTYPE_STR: &'static str = "uint";
}

impl CDatatype for i8 {
    const C_DTYPE_STR: &'static str = "char";
}

impl CDatatype for u8 {
    // TODO: different types for cuda and opencl
    //"uchar"
    const C_DTYPE_STR: &'static str = "unsigned char";
}

impl CDatatype for i16 {
    const C_DTYPE_STR: &'static str = "short";
}
impl CDatatype for u16 {
    const C_DTYPE_STR: &'static str = "ushort";
}

impl CDatatype for i64 {
    const C_DTYPE_STR: &'static str = "long";
}

impl CDatatype for u64 {
    const C_DTYPE_STR: &'static str = "ulong";
}

#[cfg(feature = "half")]
impl CDatatype for half::f16 {
    const C_DTYPE_STR: &'static str = "half";
}

// TODO: this is not bf16 - cuda and opencl name mismatch!
#[cfg(feature = "half")]
impl CDatatype for half::bf16 {
    const C_DTYPE_STR: &'static str = "half";
}
