use core::{ops::BitXor, panic::Location};

#[derive(Default)]
pub struct LocationHasher {
    hash: u64,
}

const K: u64 = 0x517cc1b727220a95;

impl std::hash::Hasher for LocationHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    #[inline]
    fn write(&mut self, _bytes: &[u8]) {
        unimplemented!("LocationHasher only hashes u64, (u32 and usize as u64 cast).")
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.hash = self.hash.rotate_left(5).bitxor(i).wrapping_mul(K);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.write_u64(i as u64);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
}

#[derive(Debug, Clone, Copy, Eq)]
pub struct HashLocation<'a> {
    file: &'a str,
    line: u32,
    col: u32,
}

impl PartialEq for HashLocation<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // filename pointer is actually actually unique, then this works (added units tests to check this... still not sure)
        if self.file.as_ptr() != other.file.as_ptr() {
            return false;
        }
        self.line == self.line && self.col == self.col
    }
}

impl<'a> std::hash::Hash for HashLocation<'a> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.file.as_ptr().hash(state);
        let line_col = (self.line as u64) << 9 | self.col as u64;
        line_col.hash(state);
    }
}

impl<'a> From<&'a Location<'a>> for HashLocation<'a> {
    #[inline]
    fn from(loc: &'a Location<'a>) -> Self {
        Self {
            file: loc.file(),
            line: loc.line(),
            col: loc.column(),
        }
    }
}
