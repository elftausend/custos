pub trait MatchesType {
    fn matches_storage_type<T: AsType>(&self) -> Result<(), String>;
}

pub trait AsType {
    const TYPE: Type;
}

impl AsType for u8 {
    const TYPE: Type = Type::U8;
}

pub enum Type {
    U8,
    U32,
    I64,
    BF16,
    F16,
    F32,
    F64,
}
