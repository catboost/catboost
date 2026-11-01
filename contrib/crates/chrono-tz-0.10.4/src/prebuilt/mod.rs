#![allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals
)]

#[cfg(not(any(feature = "case-insensitive", feature = "filter-by-regex")))]
pub(crate) mod directory;
#[cfg(not(any(feature = "case-insensitive", feature = "filter-by-regex")))]
#[rustfmt::skip]
pub(crate) mod timezones;
