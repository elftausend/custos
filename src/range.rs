use crate::Cursor;
use core::ops::{Range, RangeInclusive};

pub struct CursorRange<'a, D> {
    pub start: usize,
    pub end: usize,
    pub device: &'a D,
}

pub struct CursorRangeIter<'a, D> {
    range: CursorRange<'a, D>,
    previous_cursor: usize,
}

impl<'a, D: Cursor> Iterator for CursorRangeIter<'a, D> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.range.start >= self.range.end {
            return None;
        }
        unsafe {
            self.range.device.set_cursor(self.previous_cursor);
        }
        let epoch = self.range.start;
        self.range.start += 1;
        Some(epoch)
    }
}

impl<'a, D: Cursor> IntoIterator for CursorRange<'a, D> {
    type Item = usize;
    type IntoIter = CursorRangeIter<'a, D>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        CursorRangeIter {
            previous_cursor: self.device.cursor(),
            range: self,
        }
    }
}

/// Converts ranges into a start and end index.
pub trait AsRange {
    /// Returns the start index of the range.
    fn start(&self) -> usize;
    /// Returns the end index of the range.
    fn end(&self) -> usize;
}

impl AsRange for Range<usize> {
    #[inline]
    fn start(&self) -> usize {
        self.start
    }

    #[inline]
    fn end(&self) -> usize {
        self.end
    }
}

impl AsRange for RangeInclusive<usize> {
    #[inline]
    fn start(&self) -> usize {
        *self.start()
    }

    #[inline]
    fn end(&self) -> usize {
        *self.end() + 1
    }
}

impl AsRange for usize {
    #[inline]
    fn start(&self) -> usize {
        0
    }

    #[inline]
    fn end(&self) -> usize {
        *self
    }
}

impl AsRange for (usize, usize) {
    #[inline]
    fn start(&self) -> usize {
        self.0
    }

    #[inline]
    fn end(&self) -> usize {
        self.1
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cpu")]
    #[cfg(feature = "cached")]
    #[test]
    fn test_cursor_range() {
        use crate::{Base, Cached, Cursor, CPU};

        let device = CPU::<Cached<Base>>::new();
        for _ in device.range(10) {
            assert_eq!(device.cursor(), 0);
            unsafe { device.bump_cursor() };

            assert_eq!(device.cursor(), 1);

            for _ in device.range(20) {
                unsafe { device.bump_cursor() };
                unsafe { device.bump_cursor() };
                assert_eq!(device.cursor(), 3);
            }

            assert_eq!(device.cursor(), 3);
            unsafe { device.bump_cursor() };
            assert_eq!(device.cursor(), 4);
        }
    }

    #[cfg(feature = "cpu")]
    #[cfg(feature = "cached")]
    #[test]
    fn test_cursor_range_pre_bumped() {
        use crate::{Base, Cached, Cursor, CPU};

        let device = CPU::<Cached<Base>>::new();

        unsafe { device.bump_cursor() };
        unsafe { device.bump_cursor() };

        for _ in device.range(10) {
            assert_eq!(device.cursor(), 2);
            unsafe { device.bump_cursor() };

            assert_eq!(device.cursor(), 3);

            for _ in device.range(20) {
                unsafe { device.bump_cursor() };
                unsafe { device.bump_cursor() };
                assert_eq!(device.cursor(), 5);
            }

            assert_eq!(device.cursor(), 5);
            unsafe { device.bump_cursor() };
            assert_eq!(device.cursor(), 6);
        }
    }
}
