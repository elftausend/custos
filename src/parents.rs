use crate::{Buffer, Device, HasId, Id, Shape};

pub trait Parents<const N: usize> {
    fn ids(&self) -> [Id; N];
}

impl Parents<0> for () {
    #[inline]
    fn ids(&self) -> [Id; 0] {
        []
    }
}

impl<T: HasId> Parents<1> for T {
    #[inline]
    fn ids(&self) -> [Id; 1] {
        [self.id()]
    }
}
impl<T: HasId, T1: HasId> Parents<2> for (T, T1) {
    #[inline]
    fn ids(&self) -> [Id; 2] {
        let (first, second) = self;
        [first.id(), second.id()]
    }
}

impl Parents<3> for (Id, Id, Id) {
    #[inline]
    fn ids(&self) -> [Id; 3] {
        [self.0, self.1, self.2]
    }
}

impl<T, D: Device, S: Shape, T1, D1: Device, S1: Shape, T2, D2: Device, S2: Shape> Parents<3>
    for (
        &Buffer<'_, T, D, S>,
        &Buffer<'_, T1, D1, S1>,
        &Buffer<'_, T2, D2, S2>,
    )
{
    #[inline]
    fn ids(&self) -> [Id; 3] {
        let (buf, buf1, buf2) = self;
        [buf.id(), buf1.id(), buf2.id()]
    }
}

impl<T: HasId + Copy, const N: usize> Parents<N> for [T; N] {
    fn ids(&self) -> [Id; N] {
        self.map(|buf| buf.id())
    }
}
