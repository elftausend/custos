use crate::{Alloc, BufFlag, Buffer, GraphReturn};

use super::static_cpu;

impl<'a, A> FromIterator<A> for Buffer<'a, A>
where
    A: Clone + Default,
{
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        let device = static_cpu();
        let from_iter = Vec::from_iter(iter);

        Buffer {
            len: from_iter.len(),
            node: device.graph().add_leaf(from_iter.len()),
            ptr: device.alloc_with_vec(from_iter),
            device: Some(device),
            flag: BufFlag::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{buf, Buffer};

    #[test]
    fn test_from_iter() {
        let buf = Buffer::from_iter(0..10);
        assert_eq!(buf.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    }

    #[test]
    fn test_collect() {
        let buf = (0..5).into_iter().collect::<Buffer<i32>>();

        assert_eq!(buf.read(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_macro_filling() {
        let buf = buf![2.; 10];
        assert_eq!(buf.as_slice(), &[2.; 10]);
    }

    #[test]
    fn test_macro_from_slice() {
        let buf = buf![5, 3, 2, 6, 2];
        assert_eq!(buf.as_slice(), &[5, 3, 2, 6, 2])
    }
}
