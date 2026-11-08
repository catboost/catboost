//! Re-export of Cow
//!
//! If feature `beef` is enabled, this will re-export [`beef::lean::Cow`][beef].
//! Otherwise, it will re-export [`std::borrow::Cow`].
//!
//! [beef]: https://docs.rs/beef/latest/beef/lean/type.Cow.html
#[cfg(not(feature = "beef"))]
pub use std::borrow::Cow;

#[cfg(feature = "beef")]
pub use beef::lean::Cow;
