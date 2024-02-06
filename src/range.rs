use crate::Cursor;
use core::ops::{Range, RangeInclusive};

pub struct CursorRange<'a, D> {
    start: usize,
    end: usize,
    device: &'a D,
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
