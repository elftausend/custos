use core::ops::{Range, RangeInclusive};

// pub struct ExecRange<'a, D: Cursor> {
//     start: usize,
//     end: usize,
//     device: &'a Dev
// }

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
