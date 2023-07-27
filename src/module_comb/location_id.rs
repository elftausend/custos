
macro_rules! location_id {
    ($file:expr, $line:expr, $column:expr) => {{
        let hash_location = $crate::module_comb::HashLocation {
            file: $file,
            line: $line,
            col: $column,
        };
        let mut hasher = $crate::module_comb::LocationHasher::default();

        use std::hash::{Hash, Hasher};

        hash_location.hash(&mut hasher);

        $crate::module_comb::LocationId {
            id: hasher.finish() as usize,
        }
    }};
    () => {
        location_id!(file!(), line!(), column!())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocationId {
    pub id: usize,
}

impl LocationId {
    #[inline]
    #[track_caller]
    pub fn new() -> Self {        
        let location = core::panic::Location::caller();
        location_id!(location.file(), location.line(), location.column())
    }
}

impl Default for LocationId {
    #[inline]
    #[track_caller]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::module_comb::LocationId;

    fn untracked_location() -> LocationId {
        LocationId::default()
    }

    #[track_caller]
    fn tracked_location() -> LocationId {
        LocationId::new()
    }

    #[test]
    fn test_untracked_location_id_same_id() {
        let loc1 = untracked_location();
        let loc2 = untracked_location();
        assert_eq!(loc1, loc2);
    }

    #[test]
    fn test_tracked_location_id_different_id() {
        let loc1 = tracked_location();
        let loc2 = tracked_location();
        assert_ne!(loc1, loc2);
    }
}
