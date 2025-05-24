use crate::{Buffer, Unit};

use super::StaticDevice;

impl<'a, A, D> FromIterator<A> for Buffer<'a, A, D>
where
    A: Unit + Clone + Default,
    D: StaticDevice + 'static,
    Buffer<'a, A, D>: From<(&'a D, Vec<A>)>,
{
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        let device = D::as_static();
        let from_iter = Vec::from_iter(iter);
        Buffer::from((device, from_iter))
    }
}

#[cfg(test)]
mod tests {
    use crate::static_api::CpuBuffer;

    #[test]
    fn test_from_iter() {
        let buf: CpuBuffer<i32> = FromIterator::from_iter(0..10);
        assert_eq!(buf.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    }

    #[test]
    fn test_collect() {
        let buf = (0..5).into_iter().collect::<CpuBuffer<i32>>();
        assert_eq!(buf.read(), &[0, 1, 2, 3, 4]);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_from_iter_cl() {
        use crate::{static_api::Mods, Buffer, OpenCL};

        let buf: Buffer<i32, OpenCL<Mods<OpenCL>>> = FromIterator::from_iter(0..10);
        assert_eq!(buf.read(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_collect_cl() {
        use crate::{static_api::Mods, Buffer, OpenCL};

        let buf = (0..5).into_iter().collect::<Buffer<i32, OpenCL<Mods<OpenCL>>>>();
        assert_eq!(buf.read(), &[0, 1, 2, 3, 4]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_from_iter_cuda() {
        use crate::{static_api::Mods, Buffer, CUDA};

        let buf: Buffer<i32, CUDA<Mods<CUDA>>> = FromIterator::from_iter(0..10);
        assert_eq!(buf.read(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_collect_cuda() {
        use crate::{static_api::Mods, Buffer, CUDA};

        let buf = (0..5).into_iter().collect::<Buffer<i32, CUDA<Mods<CUDA>>>>();
        assert_eq!(buf.read(), &[0, 1, 2, 3, 4]);
    }
}
