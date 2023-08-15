use custos::prelude::*;
use std::mem::size_of;

#[cfg(feature = "cuda")]
#[test]
fn test_read_cuda() -> custos::Result<()> {
    let device = CUDA::<Base>::new(0)?;
    let a = Buffer::from((&device, [3., 1., 3., 4.]));
    let read = device.read(&a);
    assert_eq!(vec![3., 1., 3., 4.,], read);
    Ok(())
}

fn slice_u8_cast<T>(input: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * size_of::<T>()) }
}

#[test]
fn test_slice_cast() {
    let slice = &[5., 2., 7., 2.];
    let u8_slice = slice_u8_cast(slice);

    assert_eq!(
        [
            0, 0, 0, 0, 0, 0, 20, 64, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 28, 64, 0, 0, 0,
            0, 0, 0, 0, 64
        ],
        u8_slice
    )
}
