use custos::{Buffer, Device, Shape};

pub trait ElementWise<'a, T, D: Device, S: Shape = ()>: Device {
    fn add(
        &'a self,
        lhs: &Buffer<'a, T, D, S>,
        rhs: &Buffer<'a, T, D, S>,
    ) -> Buffer<'a, T, Self, S>;
    fn mul(
        &'a self,
        lhs: &Buffer<'a, T, D, S>,
        rhs: &Buffer<'a, T, D, S>,
    ) -> Buffer<'a, T, Self, S>;
}

pub trait ElementWiseBuf<T, D: Device> {
    fn add(&self, rhs: &Buffer<T, D>) -> Buffer<T, D>;
}

impl<'a, T, D: Device + ElementWise<'a, T, D>> ElementWiseBuf<T, D> for Buffer<'a, T, D> {
    fn add(&self, rhs: &Buffer<T, D>) -> Buffer<T, D> {
        todo!()
        //self.device().add(self, rhs)
    }
}

pub trait Add2<Rhs = Self> {
    type Output;

    fn add(self, rhs: Rhs) -> Self::Output;
}

impl<'a, T, D: Device + ElementWise<'a, T, D>> Add2 for &Buffer<'a, T, D> {
    type Output = Buffer<'a, T, D>;

    fn add(self, rhs: Self) -> Self::Output {
        self.device().add(self, rhs)
    }
}

pub struct Matrix<'a, T, D: Device> {
    buf: Buffer<'a, T, D>,
}

impl<'a, T, D: Device> Matrix<'a, T, D> {
    #[inline]
    pub fn device(&self) -> &'a D {
        self.buf.device()
    }
}

impl<'a, T, D: Device + ElementWise<'a, T, D>> Add2 for &Matrix<'a, T, D> {
    type Output = Matrix<'a, T, D>;

    fn add(self, rhs: Self) -> Self::Output {
        Matrix {
            buf: self.device().add(&self.buf, &rhs.buf),
        }
    }
}
