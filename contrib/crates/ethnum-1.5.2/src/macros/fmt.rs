//! Module containing macros for implementing `core::fmt` traits.

macro_rules! impl_fmt {
    (impl Fmt for $int:ident;) => {
        __impl_fmt_base! { Binary   for $int }
        __impl_fmt_base! { Octal    for $int }
        __impl_fmt_base! { LowerHex for $int }
        __impl_fmt_base! { UpperHex for $int }

        impl ::core::fmt::Debug for $int {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                // NOTE: Work around `Formatter::debug_{lower,upper}_hex` being private
                // and not stabilized.
                #[allow(deprecated)]
                let flags = f.flags();
                const DEBUG_LOWER_HEX: u32 = 1 << 4;
                const DEBUG_UPPER_HEX: u32 = 1 << 5;

                if flags & DEBUG_LOWER_HEX != 0 {
                    ::core::fmt::LowerHex::fmt(self, f)
                } else if flags & DEBUG_UPPER_HEX != 0 {
                    ::core::fmt::UpperHex::fmt(self, f)
                } else {
                    ::core::fmt::Display::fmt(self, f)
                }
            }
        }

        impl ::core::fmt::Display for $int {
            #[allow(unused_comparisons, unused_imports)]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                use $crate::uint::AsU256;

                let is_nonnegative = *self >= 0;
                let n = if is_nonnegative {
                    self.as_u256()
                } else {
                    // convert the negative num to positive by summing 1 to it's 2 complement
                    (!self.as_u256()).wrapping_add($crate::uint::U256::ONE)
                };
                $crate::fmt::fmt_u256(n, is_nonnegative, f)
            }
        }

        impl ::core::fmt::LowerExp for $int {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                // TODO(nlordell): Ideally this should be implemented similarly
                // to the primitive integer types as seen here:
                // https://doc.rust-lang.org/src/core/fmt/num.rs.html#274
                // Unfortunately, just porting this implementation is not
                // possible as it requires private standard library items. For
                // now, just convert to a `f64` as an approximation.
                ::core::fmt::LowerExp::fmt(&self.as_f64(), f)
            }
        }

        impl ::core::fmt::UpperExp for $int {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::fmt::UpperExp::fmt(&self.as_f64(), f)
            }
        }
    };
}

macro_rules! __impl_fmt_base {
    ($base:ident for $int:ident) => {
        impl ::core::fmt::$base for $int {
            #[allow(unused_imports)]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                use $crate::{fmt::GenericRadix, uint::AsU256};
                let (abs, is_nonnegative) = if *self < 0 && f.sign_minus() {
                    // NOTE(nlordell): This is non-standard break from the Rust
                    // standard integer types, but allows `format!("{val:-#x")`
                    // notation for formating a number as `-0x...` (and in
                    // in general prefix with a `-` sign for negative numbers
                    // with radix formatting.
                    (self.wrapping_neg(), false)
                } else {
                    (*self, true)
                };
                $crate::fmt::$base.fmt_u256(abs.as_u256(), is_nonnegative, f)
            }
        }
    };
}
