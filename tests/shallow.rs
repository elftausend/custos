use custos::{Buffer, CPU};

#[test]
fn test_shallow_buf_copy() {
    let device = CPU::new();

    let buf = Buffer::from((&device, [1, 2, 3, 4, 5]));
    let mut owned = unsafe { buf.shallow() };
    owned[0] = 101;

    assert_eq!(buf.as_slice(), &[101, 2, 3, 4, 5]);
}

#[cfg(feature = "realloc")]
#[test]
fn test_shallow_buf_realloc() {
    let device = CPU::new();

    let buf = Buffer::from((&device, [1, 2, 3, 4, 5]));
    let mut cloned = unsafe { buf.shallow_or_clone() };
    cloned[0] = 101;

    assert_eq!(buf.as_slice(), &[1, 2, 3, 4, 5]);
    assert_eq!(cloned.as_slice(), &[101, 2, 3, 4, 5]);
}

#[cfg(not(feature = "realloc"))]
#[test]
fn test_shallow_buf_realloc() {
    let device = CPU::new();

    let buf = Buffer::from((&device, [1, 2, 3, 4, 5]));
    let mut cloned = unsafe { buf.shallow_or_clone() };
    cloned[0] = 101;

    assert_eq!(buf.as_slice(), &[101, 2, 3, 4, 5]);
}
