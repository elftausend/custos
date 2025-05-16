use crate::Id;

#[derive(Debug, Default)]
pub enum MaybeData<Data> {
    Data(Data),
    Id(Id),
    #[default]
    None,
}

impl<Data> MaybeData<Data> {
    #[inline]
    pub fn data(&self) -> Option<&Data> {
        match self {
            MaybeData::Data(data) => Some(data),
            MaybeData::Id(_id) => None,
            MaybeData::None => None,
        }
    }

    #[inline]
    pub fn data_mut(&mut self) -> Option<&mut Data> {
        match self {
            MaybeData::Data(data) => Some(data),
            MaybeData::Id(_id) => None,
            MaybeData::None => None,
        }
    }

    #[inline]
    pub fn id(&self) -> Option<&Id> {
        match self {
            MaybeData::Data(_data) => None,
            MaybeData::Id(id) => Some(id),
            MaybeData::None => None,
        }
    }

    #[inline]
    pub fn id_mut(&mut self) -> Option<&mut Id> {
        match self {
            MaybeData::Data(_data) => None,
            MaybeData::Id(id) => Some(id),
            MaybeData::None => None,
        }
    }
}
