// #![cfg_attr(not(feature = "std"), no_std)]
#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg_attr(docsrs, doc(cfg(feature = "arraystring")))]
#[cfg(feature = "arraystring")]
pub mod arraystring;
pub mod option;
pub mod slice;
