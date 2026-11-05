//! Module containing macros for implementing `core::ops` traits.

macro_rules! impl_cmp {
    (
        impl Cmp for $int:ident ($prim:ident);
    ) => {
        impl PartialEq<$prim> for $int {
            #[inline]
            fn eq(&self, other: &$prim) -> bool {
                *self == $int::new(*other)
            }
        }

        impl PartialEq<$int> for $prim {
            #[inline]
            fn eq(&self, other: &$int) -> bool {
                $int::new(*self) == *other
            }
        }

        impl PartialOrd for $int {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<::core::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl PartialOrd<$prim> for $int {
            #[inline]
            fn partial_cmp(&self, rhs: &$prim) -> Option<::core::cmp::Ordering> {
                Some(self.cmp(&$int::new(*rhs)))
            }
        }

        impl PartialOrd<$int> for $prim {
            #[inline]
            fn partial_cmp(&self, rhs: &$int) -> Option<::core::cmp::Ordering> {
                Some($int::new(*self).cmp(rhs))
            }
        }
    };
}
