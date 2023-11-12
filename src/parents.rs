use crate::{HasId, Id, UpdateArg};

pub trait Parents<const N: usize>: AllParents {
    fn ids(&self) -> [Id; N];
    fn maybe_ids(&self) -> [Option<Id>; N];
}

impl Parents<0> for () {
    #[inline]
    fn ids(&self) -> [Id; 0] {
        []
    }

    #[inline]
    fn maybe_ids(&self) -> [Option<Id>; 0] {
        []
    }
}

impl AllParents for () {}
impl UpdateArg for () {
    fn update_arg(
        &mut self,
        _id: Option<crate::UniqueId>,
        _buffers: &mut HashMap<
            crate::UniqueId,
            Box<dyn core::any::Any>,
            core::hash::BuildHasherDefault<crate::NoHasher>,
        >,
    ) -> crate::Result<()> {
        Ok(())
    }
}

impl<T: HasId> Parents<1> for T {
    #[inline]
    fn ids(&self) -> [Id; 1] {
        [self.id()]
    }

    #[inline]
    fn maybe_ids(&self) -> [Option<Id>; 1] {
        [self.maybe_id()]
    }
}

impl<T: HasId> AllParents for T {}

macro_rules! impl_parents {
    ($num:expr, $($to_impl:ident),+) => {
        impl<$($to_impl: $crate::HasId, )+> Parents<$num> for ($($to_impl,)+) {
            #[inline]
            fn ids(&self) -> [Id; $num] {
                #[allow(non_snake_case)]
                let ($($to_impl,)+) = self;
                [$($to_impl.id(),)+]
            }

            #[inline]
            fn maybe_ids(&self) -> [Option<Id>; $num] {
                #[allow(non_snake_case)]
                let ($($to_impl,)+) = self;
                [$($to_impl.maybe_id(),)+]
            }
        }
        impl<$($to_impl: $crate::HasId, )+> AllParents for ($($to_impl,)+) {}

        #[cfg(not(feature = "no-std"))]
        impl<$($to_impl: $crate::UpdateArg + $crate::HasId, )+> $crate::UpdateArgs for ($($to_impl,)+) {
            fn update_args(&mut self,
                ids: &[Option<$crate::UniqueId>],
                buffers: &mut HashMap<$crate::UniqueId, Box<dyn std::any::Any>, core::hash::BuildHasherDefault<$crate::NoHasher>>)
             -> crate::Result<()>
             {
                let mut ids = ids.iter();
                #[allow(non_snake_case)]
                let ($($to_impl,)+) = self;
                $($to_impl.update_arg(*ids.next().unwrap(), buffers)?;)*
                Ok(())
            }
        }
    };
}

use std::collections::HashMap;

impl_parents!(2, T, T1);
impl_parents!(3, T, T1, T2);
impl_parents!(4, T, T1, T2, T3);
impl_parents!(5, T, T1, T2, T3, T4);
impl_parents!(6, T, T1, T2, T3, T4, T5);
impl_parents!(7, T, T1, T2, T3, T4, T5, T6);
impl_parents!(8, T, T1, T2, T3, T4, T5, T6, T7);

impl<T: HasId + Copy, const N: usize> Parents<N> for [T; N] {
    #[inline]
    fn ids(&self) -> [Id; N] {
        self.map(|buf| buf.id())
    }

    #[inline]
    fn maybe_ids(&self) -> [Option<Id>; N] {
        self.map(|buf| buf.maybe_id())
    }
}

impl<T: HasId + Copy, const N: usize> AllParents for [T; N] {}

pub trait AllParents {}
