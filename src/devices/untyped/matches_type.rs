use crate::Unit;

pub trait MatchesType {
    fn matches_storage_type<T: AsType>(&self) -> Result<(), String>;
}

pub trait AsDeviceType {
    const DEVICE_TYPE: DeviceType;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    CPU,
    CUDA,
}

impl<Mods> AsDeviceType for crate::CPU<Mods> {
    const DEVICE_TYPE: DeviceType = DeviceType::CPU;
}

#[cfg(feature = "cuda")]
impl<Mods> AsDeviceType for crate::CUDA<Mods> {
    const DEVICE_TYPE: DeviceType = DeviceType::CUDA;
}

pub trait AsType: Unit {
    const TYPE: Type;
}

impl AsType for u8 {
    const TYPE: Type = Type::U8;
}

impl AsType for u32 {
    const TYPE: Type = Type::U32;
}

impl AsType for i64 {
    const TYPE: Type = Type::I64;
}

#[cfg(feature = "half")]
impl AsType for half::bf16 {
    const TYPE: Type = Type::BF16;
}

#[cfg(feature = "half")]
impl AsType for half::f16 {
    const TYPE: Type = Type::F16;
}

impl AsType for f32 {
    const TYPE: Type = Type::F32;
}

impl AsType for f64 {
    const TYPE: Type = Type::F64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    U8,
    U32,
    I64,
    #[cfg(feature = "half")]
    BF16,
    #[cfg(feature = "half")]
    F16,
    F32,
    F64,
}
