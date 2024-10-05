#[cfg(feature = "std")]
mod span;

#[cfg(feature = "std")]
pub use span::*;

use crate::Cursor;
use core::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

pub struct CursorRange<'a, D> {
    pub start: usize,
    pub end: usize,
    pub device: &'a D,
}

pub struct CursorRangeIter<'a, D> {
    range: CursorRange<'a, D>,
    previous_cursor: usize,
}

impl<'a, D: Cursor> CursorRangeIter<'a, D> {
    #[inline]
    pub fn previous_cursor(&self) -> &usize {
        &self.previous_cursor
    }
    
    #[inline]
    pub fn cursor_range(&self) -> &CursorRange<'a, D> {
        &self.range
    }
}

impl<'a, D: Cursor> Iterator for CursorRangeIter<'a, D> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.range.start >= self.range.end {
            return None;
        }
        let epoch = self.range.start;

        unsafe {
            self.range.device.set_cursor(self.previous_cursor);
        }
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


// Implementing AsRange for standard Range (e.g., 0..10)
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

// Implementing AsRange for RangeInclusive (e.g., 0..=10)
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

// Implementing AsRange for a single usize (e.g., 10 means range 0..10)
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

// Implementing AsRange for a tuple (usize, usize) (e.g., (5, 10))
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

// Implementing AsRange for RangeTo (e.g., ..10)
impl AsRange for RangeTo<usize> {
    #[inline]
    fn start(&self) -> usize {
        0
    }

    #[inline]
    fn end(&self) -> usize {
        self.end
    }
}

// Implementing AsRange for RangeFrom (e.g., 5..)
impl AsRange for RangeFrom<usize> {
    #[inline]
    fn start(&self) -> usize {
        self.start
    }

    #[inline]
    fn end(&self) -> usize {
        usize::MAX // Unbounded range goes to maximum usize
    }
}

// Implementing AsRange for RangeFull (e.g., ..)
impl AsRange for RangeFull {
    #[inline]
    fn start(&self) -> usize {
        0
    }

    #[inline]
    fn end(&self) -> usize {
        usize::MAX // Represents the entire range
    }
}

// Implementing AsRange for RangeToInclusive (e.g., ..=10)
impl AsRange for RangeToInclusive<usize> {
    #[inline]
    fn start(&self) -> usize {
        0
    }

    #[inline]
    fn end(&self) -> usize {
        self.end + 1
    }
}


#[cfg(test)]
mod tests {
    use crate::{Base, Cached, Cursor, CPU}; // Moved shared imports to the top

    #[cfg(feature = "cpu")]
    #[cfg(feature = "cached")]
    #[test]
    fn test_cursor_range() {
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
        assert_eq!(device.cursor(), 6);

        for _ in device.range(10) {
            assert_eq!(device.cursor(), 6);
            unsafe { device.bump_cursor() };

            assert_eq!(device.cursor(), 7);

            for _ in device.range(20) {
                unsafe { device.bump_cursor() };
                unsafe { device.bump_cursor() };
                assert_eq!(device.cursor(), 9);
            }

            assert_eq!(device.cursor(), 9);
            unsafe { device.bump_cursor() };
            assert_eq!(device.cursor(), 10);
        }
    } 

    #[cfg(feature = "cpu")]
    #[cfg(feature = "cached")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_cache_span_resetting() {
        use crate::{range::SpanStorage, span}; // Additional imports for this test
        let mut span_storage = SpanStorage::default();
        let device = CPU::<Cached<Base>>::new();

        for _ in 0..10 {
            span!(device, span_storage);
            unsafe { device.bump_cursor() };
            assert_eq!(device.cursor(), 1);

            for _ in 0..20 {
                span!(device, span_storage);
                unsafe { device.bump_cursor() };
                unsafe { device.bump_cursor() };
                assert_eq!(device.cursor(), 3);
            }

            unsafe { device.bump_cursor() };
            assert_eq!(device.cursor(), 4);
        }
        assert_eq!(device.cursor(), 4);
    }

    // Additional range tests

    #[cfg(feature = "cpu")]
    #[cfg(feature = "cached")]
    #[test]
    fn test_cursor_range_inclusive() {
        let device = CPU::<Cached<Base>>::new();
        for _ in device.range(5..=10) {
            assert_eq!(device.cursor(), 0);
            unsafe { device.bump_cursor() };
            assert_eq!(device.cursor(), 1);
        }
        unsafe { device.bump_cursor() };
        assert_eq!(device.cursor(), 2);
    }

    #[cfg(feature = "cpu")]
    #[cfg(feature = "cached")]
    #[test]
    fn test_cursor_range_to() {
        let device = CPU::<Cached<Base>>::new();
        for _ in device.range(..10) {
            assert_eq!(device.cursor(), 0);
            unsafe { device.bump_cursor() };
            assert_eq!(device.cursor(), 1);
        }
        unsafe { device.bump_cursor() };
        assert_eq!(device.cursor(), 2);
    }

    #[cfg(feature = "cpu")]
    #[cfg(feature = "cached")]
    #[test]
    fn test_cursor_range_from() {
        let device = CPU::<Cached<Base>>::new();
        for _ in device.range(5..) {
            assert_eq!(device.cursor(), 0);
            unsafe { device.bump_cursor() };
            assert_eq!(device.cursor(), 1);
            break; // Ensure that I don't run into infinite loop
        }
        unsafe { device.bump_cursor() };
        assert_eq!(device.cursor(), 2);
    }

    #[cfg(feature = "cpu")]
    #[cfg(feature = "cached")]
    #[test]
    fn test_cursor_range_full() {
        let device = CPU::<Cached<Base>>::new();
        for _ in device.range(..) {
            assert_eq!(device.cursor(), 0);
            unsafe { device.bump_cursor() };
            assert_eq!(device.cursor(), 1);
            break; // Avoid infinite loop
        }
        unsafe { device.bump_cursor() };
        assert_eq!(device.cursor(), 2);
    }
}
