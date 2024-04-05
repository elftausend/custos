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

pub trait AsType {
    const TYPE: Type;
}

impl AsType for u8 {
    const TYPE: Type = Type::U8;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    U8,
    U32,
    I64,
    BF16,
    F16,
    F32,
    F64,
}
