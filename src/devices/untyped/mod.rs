pub mod storages;
mod untyped_device;

#[cfg(not(feature = "cuda"))]
mod dummy_cuda;

#[cfg(not(feature = "cuda"))]
pub use dummy_cuda::*;

mod matches_type;
mod ops;
pub use matches_type::*;

use crate::Buffer;

use self::untyped_device::Untyped;

impl<'a, T, S: crate::Shape> Buffer<'a, T, Untyped, S> {
    #[inline]
    pub fn to_typed<NT, NS>(self) -> Option<Buffer<'a, NT, Untyped, NS>>
    where
        NT: AsType,
        NS: crate::Shape,
    {
        self.data.matches_storage_type::<NT>().ok()?;
        Some(unsafe { std::mem::transmute(self) })
    }

    #[inline]
    pub fn to_untyped(self) -> Buffer<'a, (), Untyped, ()> {
        // Safety: An Untyped device buffer is shape and data type independent!
        // type Base<T, S: crate::Shape> = UntypedData; <- missing <T, S>
        // type Data<T, S: crate::Shape> = UntypedData; <|
        unsafe { std::mem::transmute(self) }
    }

    #[inline]
    pub fn as_typed<NT, NS>(&self) -> Option<&Buffer<'a, NT, Untyped, NS>>
    where
        NT: AsType,
        NS: crate::Shape,
    {
        self.data.matches_storage_type::<NT>().ok()?;
        Some(unsafe { std::mem::transmute(self) })
    }
    
    #[inline]
    pub fn as_typed_mut<NT, NS>(&mut self) -> Option<&mut Buffer<'a, NT, Untyped, NS>>
    where
        NT: AsType,
        NS: crate::Shape,
    {
        self.data.matches_storage_type::<NT>().ok()?;
        Some(unsafe { std::mem::transmute(self) })
    }

    #[inline]
    pub fn as_untyped(&self) -> &Buffer<'a, (), Untyped, ()> {
        unsafe { std::mem::transmute(self) }
    }
    
    #[inline]
    pub fn as_untyped_mut(&mut self) -> &mut Buffer<'a, (), Untyped, ()> {
        unsafe { std::mem::transmute(self) }
    }
}

#[cfg(test)]
mod tests {
    use crate::Device;

    use super::{storages::UntypedData, untyped_device::Untyped};

    #[test]
    fn test_to_untype_buf() {
        let device = Untyped::new().unwrap();
        let buf = device.buffer([1f32, 2., 3., 4.]).to_untyped();
        match &buf.data {
            UntypedData::CPU(cpu) => match cpu {
                super::storages::CpuStorage::F32(data) => {
                    assert_eq!(data.as_slice(), [1., 2., 3., 4.,])
                }
                _ => panic!(),
            },
            UntypedData::CUDA(cuda) => match cuda {
                #[cfg(feature = "cuda")]
                super::storages::CudaStorage::F32(data) => {
                    assert_eq!(data.read(), [1., 2., 3., 4.,])
                }
                _ => panic!(),
            },
        }
    }

    #[test]
    fn test_add_type_info_to_untyped() {
        let device = Untyped::new().unwrap();
        let buf = device.buffer([1f32, 2., 3., 4.]).to_untyped();
        let typed_buf = buf.to_typed::<f32, ()>().unwrap();
        assert_eq!(typed_buf.read(), [1., 2., 3., 4.]);
    }

    #[test]
    fn test_add_type_info_to_untyped_type_mismatch() {
        let device = Untyped::new().unwrap();
        let buf = device.buffer([1f32, 2., 3., 4.]).to_untyped();
        let typed_buf = buf.to_typed::<u32, ()>();
        assert!(typed_buf.is_none())
    }
    
    #[test]
    fn test_add_type_info_to_untyped_type_mismatch_as_ref() {
        let device = Untyped::new().unwrap();
        let buf = device.buffer([1f32, 2., 3., 4.]);
        let typed_buf = buf.as_typed::<u32, ()>();
        assert!(typed_buf.is_none())
    }
    
    #[test]
    fn test_add_type_info_to_untyped_ref() {
        let device = Untyped::new().unwrap();
        let buf = device.buffer([1f32, 2., 3., 4.]);
        let untyped = buf.as_untyped();
        let typed_buf = untyped.as_typed::<f32, ()>().unwrap();
        assert_eq!(typed_buf.read(), [1., 2., 3., 4.]);
    }
}
