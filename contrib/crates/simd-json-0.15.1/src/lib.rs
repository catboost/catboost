#![deny(warnings)]
#![cfg_attr(feature = "hints", feature(core_intrinsics))]
#![cfg_attr(feature = "portable", feature(portable_simd))]
#![warn(unused_extern_crates)]
#![deny(
    clippy::all,
    clippy::unwrap_used,
    clippy::unnecessary_unwrap,
    clippy::pedantic,
    missing_docs
)]
#![allow(
    clippy::module_name_repetitions,
    unused_unsafe // for nightly
)]
#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]

#[cfg(feature = "serde_impl")]
extern crate serde as serde_ext;

#[cfg(feature = "serde_impl")]
/// serde related helper functions
pub mod serde;

#[cfg(test)]
/// serde related helper functions
pub mod tests;

use crate::error::InternalError;
#[cfg(feature = "serde_impl")]
pub use crate::serde::{
    from_reader, from_slice, from_str, to_string, to_string_pretty, to_vec, to_vec_pretty,
    to_writer, to_writer_pretty,
};

/// Default trait imports;
pub mod prelude;

mod charutils;
#[macro_use]
mod macros;
mod error;
mod numberparse;
mod safer_unchecked;
mod stringparse;

use macros::static_cast_u64;
use safer_unchecked::GetSaferUnchecked;
use stage2::StackState;
use tape::Value;

mod impls;

/// Re-export of Cow
pub mod cow;

/// The maximum padding size required by any SIMD implementation
pub(crate) const SIMDJSON_PADDING: usize = 32; // take upper limit mem::size_of::<__m256i>()
/// It's 64 for all (Is this correct?)
pub(crate) const SIMDINPUT_LENGTH: usize = 64;

mod stage2;
/// simd-json JSON-DOM value
pub mod value;

use std::{alloc::dealloc, mem};
pub use value_trait::StaticNode;

pub use crate::error::{Error, ErrorType};
#[doc(inline)]
pub use crate::value::*;
pub use value_trait::ValueType;

/// simd-json Result type
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(feature = "known-key")]
mod known_key;
#[cfg(feature = "known-key")]
pub use known_key::{Error as KnownKeyError, KnownKey};

pub use crate::tape::{Node, Tape};
use std::alloc::{Layout, alloc, handle_alloc_error};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

use simdutf8::basic::imp::ChunkedUtf8Validator;

/// A struct to hold the buffers for the parser.
pub struct Buffers {
    string_buffer: Vec<u8>,
    structural_indexes: Vec<u32>,
    input_buffer: AlignedBuf,
    stage2_stack: Vec<StackState>,
}

impl Default for Buffers {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn default() -> Self {
        Self::new(128)
    }
}

impl Buffers {
    /// Create new buffer for input length.
    /// If this is too small a new buffer will be allocated, if needed during parsing.
    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[must_use]
    pub fn new(input_len: usize) -> Self {
        // this is a heuristic, it will likely be higher but it will avoid some reallocations hopefully
        let heuristic_index_cout = input_len / 128;
        Self {
            string_buffer: Vec::with_capacity(input_len + SIMDJSON_PADDING),
            structural_indexes: Vec::with_capacity(heuristic_index_cout),
            input_buffer: AlignedBuf::with_capacity(input_len + SIMDJSON_PADDING * 2),
            stage2_stack: Vec::with_capacity(heuristic_index_cout),
        }
    }
}

/// Creates a tape from the input for later consumption
/// # Errors
///
/// Will return `Err` if `s` is invalid JSON.
#[cfg_attr(not(feature = "no-inline"), inline)]
pub fn to_tape(s: &mut [u8]) -> Result<Tape> {
    Deserializer::from_slice(s).map(Deserializer::into_tape)
}

/// Creates a tape from the input for later consumption
/// # Errors
///
/// Will return `Err` if `s` is invalid JSON.
#[cfg_attr(not(feature = "no-inline"), inline)]
pub fn to_tape_with_buffers<'de>(s: &'de mut [u8], buffers: &mut Buffers) -> Result<Tape<'de>> {
    Deserializer::from_slice_with_buffers(s, buffers).map(Deserializer::into_tape)
}

/// Fills a already existing tape from the input for later consumption
/// # Errors
///
/// Will return `Err` if `s` is invalid JSON.
#[cfg_attr(not(feature = "no-inline"), inline)]
pub fn fill_tape<'de>(s: &'de mut [u8], buffers: &mut Buffers, tape: &mut Tape<'de>) -> Result<()> {
    tape.0.clear();
    Deserializer::fill_tape(s, buffers, &mut tape.0)
}

pub(crate) trait Stage1Parse {
    type Utf8Validator: ChunkedUtf8Validator;
    type SimdRepresentation;

    unsafe fn new(ptr: &[u8]) -> Self;

    unsafe fn compute_quote_mask(quote_bits: u64) -> u64;

    unsafe fn cmp_mask_against_input(&self, m: u8) -> u64;

    unsafe fn unsigned_lteq_against_input(&self, maxval: Self::SimdRepresentation) -> u64;

    unsafe fn find_whitespace_and_structurals(&self, whitespace: &mut u64, structurals: &mut u64);

    unsafe fn flatten_bits(base: &mut Vec<u32>, idx: u32, bits: u64);

    // return both the quote mask (which is a half-open mask that covers the first
    // quote in an unescaped quote pair and everything in the quote pair) and the
    // quote bits, which are the simple unescaped quoted bits.
    //
    // We also update the prev_iter_inside_quote value to tell the next iteration
    // whether we finished the final iteration inside a quote pair; if so, this
    // inverts our behavior of whether we're inside quotes for the next iteration.
    //
    // Note that we don't do any error checking to see if we have backslash
    // sequences outside quotes; these
    // backslash sequences (of any length) will be detected elsewhere.
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn find_quote_mask_and_bits(
        &self,
        odd_ends: u64,
        prev_iter_inside_quote: &mut u64,
        quote_bits: &mut u64,
        error_mask: &mut u64,
    ) -> u64 {
        unsafe {
            *quote_bits = self.cmp_mask_against_input(b'"');
            *quote_bits &= !odd_ends;
            // remove from the valid quoted region the unescaped characters.
            let mut quote_mask: u64 = Self::compute_quote_mask(*quote_bits);
            quote_mask ^= *prev_iter_inside_quote;
            // All Unicode characters may be placed within the
            // quotation marks, except for the characters that MUST be escaped:
            // quotation mark, reverse solidus, and the control characters (U+0000
            //through U+001F).
            // https://tools.ietf.org/html/rfc8259
            let unescaped: u64 = self.unsigned_lteq_against_input(Self::fill_s8(0x1F));
            *error_mask |= quote_mask & unescaped;
            // right shift of a signed value expected to be well-defined and standard
            // compliant as of C++20,
            // John Regher from Utah U. says this is fine code
            *prev_iter_inside_quote = static_cast_u64!(static_cast_i64!(quote_mask) >> 63);
            quote_mask
        }
    }

    // return a bitvector indicating where we have characters that end an odd-length
    // sequence of backslashes (and thus change the behavior of the next character
    // to follow). A even-length sequence of backslashes, and, for that matter, the
    // largest even-length prefix of our odd-length sequence of backslashes, simply
    // modify the behavior of the backslashes themselves.
    // We also update the prev_iter_ends_odd_backslash reference parameter to
    // indicate whether we end an iteration on an odd-length sequence of
    // backslashes, which modifies our subsequent search for odd-length
    // sequences of backslashes in an obvious way.
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn find_odd_backslash_sequences(&self, prev_iter_ends_odd_backslash: &mut u64) -> u64 {
        const EVEN_BITS: u64 = 0x5555_5555_5555_5555;
        const ODD_BITS: u64 = !EVEN_BITS;

        let bs_bits: u64 = unsafe { self.cmp_mask_against_input(b'\\') };
        let start_edges: u64 = bs_bits & !(bs_bits << 1);
        // flip lowest if we have an odd-length run at the end of the prior
        // iteration
        let even_start_mask: u64 = EVEN_BITS ^ *prev_iter_ends_odd_backslash;
        let even_starts: u64 = start_edges & even_start_mask;
        let odd_starts: u64 = start_edges & !even_start_mask;
        let even_carries: u64 = bs_bits.wrapping_add(even_starts);

        // must record the carry-out of our odd-carries out of bit 63; this
        // indicates whether the sense of any edge going to the next iteration
        // should be flipped
        let (mut odd_carries, iter_ends_odd_backslash) = bs_bits.overflowing_add(odd_starts);

        odd_carries |= *prev_iter_ends_odd_backslash;
        // push in bit zero as a potential end
        // if we had an odd-numbered run at the
        // end of the previous iteration
        *prev_iter_ends_odd_backslash = u64::from(iter_ends_odd_backslash);
        let even_carry_ends: u64 = even_carries & !bs_bits;
        let odd_carry_ends: u64 = odd_carries & !bs_bits;
        let even_start_odd_end: u64 = even_carry_ends & ODD_BITS;
        let odd_start_even_end: u64 = odd_carry_ends & EVEN_BITS;
        let odd_ends: u64 = even_start_odd_end | odd_start_even_end;
        odd_ends
    }

    // return a updated structural bit vector with quoted contents cleared out and
    // pseudo-structural characters added to the mask
    // updates prev_iter_ends_pseudo_pred which tells us whether the previous
    // iteration ended on a whitespace or a structural character (which means that
    // the next iteration
    // will have a pseudo-structural character at its start)
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn finalize_structurals(
        mut structurals: u64,
        whitespace: u64,
        quote_mask: u64,
        quote_bits: u64,
        prev_iter_ends_pseudo_pred: &mut u64,
    ) -> u64 {
        // mask off anything inside quotes
        structurals &= !quote_mask;
        // add the real quote bits back into our bitmask as well, so we can
        // quickly traverse the strings we've spent all this trouble gathering
        structurals |= quote_bits;
        // Now, establish "pseudo-structural characters". These are non-whitespace
        // characters that are (a) outside quotes and (b) have a predecessor that's
        // either whitespace or a structural character. This means that subsequent
        // passes will get a chance to encounter the first character of every string
        // of non-whitespace and, if we're parsing an atom like true/false/null or a
        // number we can stop at the first whitespace or structural character
        // following it.

        // a qualified predecessor is something that can happen 1 position before an
        // pseudo-structural character
        let pseudo_pred: u64 = structurals | whitespace;

        let shifted_pseudo_pred: u64 = (pseudo_pred << 1) | *prev_iter_ends_pseudo_pred;
        *prev_iter_ends_pseudo_pred = pseudo_pred >> 63;
        let pseudo_structurals: u64 = shifted_pseudo_pred & (!whitespace) & (!quote_mask);
        structurals |= pseudo_structurals;

        // now, we've used our close quotes all we need to. So let's switch them off
        // they will be off in the quote mask and on in quote bits.
        structurals &= !(quote_bits & !quote_mask);
        structurals
    }

    unsafe fn fill_s8(n: i8) -> Self::SimdRepresentation;
}

/// Deserializer struct to deserialize a JSON
#[derive(Debug)]
pub struct Deserializer<'de> {
    // Note: we use the 2nd part as both index and length since only one is ever
    // used (array / object use len) everything else uses idx
    pub(crate) tape: Vec<Node<'de>>,
    idx: usize,
}

// architecture dependant parse_str

#[derive(Debug, Clone, Copy)]
pub(crate) struct SillyWrapper<'de> {
    input: *mut u8,
    _marker: std::marker::PhantomData<&'de ()>,
}

impl From<*mut u8> for SillyWrapper<'_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn from(input: *mut u8) -> Self {
        Self {
            input,
            _marker: std::marker::PhantomData,
        }
    }
}

#[cfg(all(
    feature = "runtime-detection",
    any(target_arch = "x86_64", target_arch = "x86"),
))] // The runtime detection code is inspired from simdutf8's implementation
type FnRaw = *mut ();
#[cfg(all(
    feature = "runtime-detection",
    any(target_arch = "x86_64", target_arch = "x86"),
))]
type ParseStrFn = for<'invoke, 'de> unsafe fn(
    SillyWrapper<'de>,
    &'invoke [u8],
    &'invoke mut [u8],
    usize,
) -> std::result::Result<&'de str, error::Error>;
#[cfg(all(
    feature = "runtime-detection",
    any(target_arch = "x86_64", target_arch = "x86"),
))]
type FindStructuralBitsFn = unsafe fn(
    input: &[u8],
    structural_indexes: &mut Vec<u32>,
) -> std::result::Result<(), ErrorType>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Supported implementations
pub enum Implementation {
    /// Rust native implementation
    Native,
    /// Rust native implementation with using [`std::simd`]
    StdSimd,
    /// SSE4.2 implementation
    SSE42,
    /// AVX2 implementation
    AVX2,
    /// ARM NEON implementation
    NEON,
    /// WEBASM SIMD128 implementation
    SIMD128,
}

impl std::fmt::Display for Implementation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Implementation::Native => write!(f, "Rust Native"),
            Implementation::StdSimd => write!(f, "std::simd"),
            Implementation::SSE42 => write!(f, "SSE42"),
            Implementation::AVX2 => write!(f, "AVX2"),
            Implementation::NEON => write!(f, "NEON"),
            Implementation::SIMD128 => write!(f, "SIMD128"),
        }
    }
}

impl Deserializer<'_> {
    /// returns the algorithm / architecture used by the deserializer
    #[cfg(all(
        feature = "runtime-detection",
        any(target_arch = "x86_64", target_arch = "x86"),
    ))]
    #[must_use]
    pub fn algorithm() -> Implementation {
        if std::is_x86_feature_detected!("avx2") {
            Implementation::AVX2
        } else if std::is_x86_feature_detected!("sse4.2") {
            Implementation::SSE42
        } else {
            #[cfg(feature = "portable")]
            let r = Implementation::StdSimd;
            #[cfg(not(feature = "portable"))]
            let r = Implementation::Native;
            r
        }
    }
    #[cfg(not(any(
        all(
            feature = "runtime-detection",
            any(target_arch = "x86_64", target_arch = "x86")
        ),
        feature = "portable",
        target_feature = "avx2",
        target_feature = "sse4.2",
        target_feature = "simd128",
        target_arch = "aarch64",
    )))]
    /// returns the algorithm / architecture used by the deserializer
    #[must_use]
    pub fn algorithm() -> Implementation {
        Implementation::Native
    }
    #[cfg(all(feature = "portable", not(feature = "runtime-detection")))]
    /// returns the algorithm / architecture used by the deserializer
    #[must_use]
    pub fn algorithm() -> Implementation {
        Implementation::StdSimd
    }

    #[cfg(all(
        target_feature = "avx2",
        not(feature = "portable"),
        not(feature = "runtime-detection"),
    ))]
    /// returns the algorithm / architecture used by the deserializer
    #[must_use]
    pub fn algorithm() -> Implementation {
        Implementation::AVX2
    }

    #[cfg(all(
        target_feature = "sse4.2",
        not(target_feature = "avx2"),
        not(feature = "runtime-detection"),
        not(feature = "portable"),
    ))]
    /// returns the algorithm / architecture used by the deserializer
    #[must_use]
    pub fn algorithm() -> Implementation {
        Implementation::SSE42
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "portable")))]
    /// returns the algorithm / architecture used by the deserializer
    #[must_use]
    pub fn algorithm() -> Implementation {
        Implementation::NEON
    }

    #[cfg(all(target_feature = "simd128", not(feature = "portable")))]
    /// returns the algorithm / architecture used by the deserializer
    #[must_use]
    pub fn algorithm() -> Implementation {
        Implementation::SIMD128
    }
}

impl<'de> Deserializer<'de> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[cfg(all(
        feature = "runtime-detection",
        any(target_arch = "x86_64", target_arch = "x86"),
    ))]
    pub(crate) unsafe fn parse_str_<'invoke>(
        input: *mut u8,
        data: &'invoke [u8],
        buffer: &'invoke mut [u8],
        idx: usize,
    ) -> Result<&'de str>
    where
        'de: 'invoke,
    {
        unsafe {
            use std::sync::atomic::{AtomicPtr, Ordering};

            static FN: AtomicPtr<()> = AtomicPtr::new(get_fastest as FnRaw);

            #[cfg_attr(not(feature = "no-inline"), inline)]
            fn get_fastest_available_implementation() -> ParseStrFn {
                if std::is_x86_feature_detected!("avx2") {
                    impls::avx2::parse_str
                } else if std::is_x86_feature_detected!("sse4.2") {
                    impls::sse42::parse_str
                } else {
                    #[cfg(feature = "portable")]
                    let r = impls::portable::parse_str;
                    #[cfg(not(feature = "portable"))]
                    let r = impls::native::parse_str;
                    r
                }
            }

            #[cfg_attr(not(feature = "no-inline"), inline)]
            unsafe fn get_fastest<'invoke, 'de>(
                input: SillyWrapper<'de>,
                data: &'invoke [u8],
                buffer: &'invoke mut [u8],
                idx: usize,
            ) -> core::result::Result<&'de str, error::Error>
            where
                'de: 'invoke,
            {
                unsafe {
                    let fun = get_fastest_available_implementation();
                    FN.store(fun as FnRaw, Ordering::Relaxed);
                    (fun)(input, data, buffer, idx)
                }
            }

            let input: SillyWrapper<'de> = SillyWrapper::from(input);
            let fun = FN.load(Ordering::Relaxed);
            mem::transmute::<FnRaw, ParseStrFn>(fun)(input, data, buffer, idx)
        }
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[cfg(not(any(
        all(
            feature = "runtime-detection",
            any(target_arch = "x86_64", target_arch = "x86")
        ),
        feature = "portable",
        target_feature = "avx2",
        target_feature = "sse4.2",
        target_feature = "simd128",
        target_arch = "aarch64",
    )))]
    pub(crate) unsafe fn parse_str_<'invoke>(
        input: *mut u8,
        data: &'invoke [u8],
        buffer: &'invoke mut [u8],
        idx: usize,
    ) -> Result<&'de str>
    where
        'de: 'invoke,
    {
        let input: SillyWrapper<'de> = SillyWrapper::from(input);
        unsafe { impls::native::parse_str(input, data, buffer, idx) }
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[cfg(all(feature = "portable", not(feature = "runtime-detection")))]
    pub(crate) unsafe fn parse_str_<'invoke>(
        input: *mut u8,
        data: &'invoke [u8],
        buffer: &'invoke mut [u8],
        idx: usize,
    ) -> Result<&'de str>
    where
        'de: 'invoke,
    {
        let input: SillyWrapper<'de> = SillyWrapper::from(input);
        impls::portable::parse_str(input, data, buffer, idx)
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[cfg(all(
        target_feature = "avx2",
        not(feature = "portable"),
        not(feature = "runtime-detection"),
    ))]
    pub(crate) unsafe fn parse_str_<'invoke>(
        input: *mut u8,
        data: &'invoke [u8],
        buffer: &'invoke mut [u8],
        idx: usize,
    ) -> Result<&'de str> {
        let input: SillyWrapper<'de> = SillyWrapper::from(input);
        unsafe { impls::avx2::parse_str(input, data, buffer, idx) }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[cfg(all(
        target_feature = "sse4.2",
        not(target_feature = "avx2"),
        not(feature = "runtime-detection"),
        not(feature = "portable"),
    ))]
    pub(crate) unsafe fn parse_str_<'invoke>(
        input: *mut u8,
        data: &'invoke [u8],
        buffer: &'invoke mut [u8],
        idx: usize,
    ) -> Result<&'de str> {
        let input: SillyWrapper<'de> = SillyWrapper::from(input);
        unsafe { impls::sse42::parse_str(input, data, buffer, idx) }
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[cfg(all(target_arch = "aarch64", not(feature = "portable")))]
    pub(crate) unsafe fn parse_str_<'invoke>(
        input: *mut u8,
        data: &'invoke [u8],
        buffer: &'invoke mut [u8],
        idx: usize,
    ) -> Result<&'de str> {
        let input: SillyWrapper = SillyWrapper::from(input);
        impls::neon::parse_str(input, data, buffer, idx)
    }
    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[cfg(all(target_feature = "simd128", not(feature = "portable")))]
    pub(crate) unsafe fn parse_str_<'invoke>(
        input: *mut u8,
        data: &'invoke [u8],
        buffer: &'invoke mut [u8],
        idx: usize,
    ) -> Result<&'de str> {
        let input: SillyWrapper<'de> = SillyWrapper::from(input);
        impls::simd128::parse_str(input, data, buffer, idx)
    }
}

/// architecture dependant `find_structural_bits`
impl Deserializer<'_> {
    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[cfg(all(
        feature = "runtime-detection",
        any(target_arch = "x86_64", target_arch = "x86"),
    ))]
    pub(crate) unsafe fn find_structural_bits(
        input: &[u8],
        structural_indexes: &mut Vec<u32>,
    ) -> std::result::Result<(), ErrorType> {
        unsafe {
            use std::sync::atomic::{AtomicPtr, Ordering};

            static FN: AtomicPtr<()> = AtomicPtr::new(get_fastest as FnRaw);

            #[cfg_attr(not(feature = "no-inline"), inline)]
            fn get_fastest_available_implementation() -> FindStructuralBitsFn {
                if std::is_x86_feature_detected!("avx2") {
                    Deserializer::_find_structural_bits::<impls::avx2::SimdInput>
                } else if std::is_x86_feature_detected!("sse4.2") {
                    Deserializer::_find_structural_bits::<impls::sse42::SimdInput>
                } else {
                    #[cfg(feature = "portable")]
                    let r = Deserializer::_find_structural_bits::<impls::portable::SimdInput>;
                    #[cfg(not(feature = "portable"))]
                    let r = Deserializer::_find_structural_bits::<impls::native::SimdInput>;
                    r
                }
            }

            #[cfg_attr(not(feature = "no-inline"), inline)]
            unsafe fn get_fastest(
                input: &[u8],
                structural_indexes: &mut Vec<u32>,
            ) -> core::result::Result<(), error::ErrorType> {
                unsafe {
                    let fun = get_fastest_available_implementation();
                    FN.store(fun as FnRaw, Ordering::Relaxed);
                    (fun)(input, structural_indexes)
                }
            }

            let fun = FN.load(Ordering::Relaxed);
            mem::transmute::<FnRaw, FindStructuralBitsFn>(fun)(input, structural_indexes)
        }
    }

    #[cfg(not(any(
        all(
            feature = "runtime-detection",
            any(target_arch = "x86_64", target_arch = "x86")
        ),
        feature = "portable",
        target_feature = "avx2",
        target_feature = "sse4.2",
        target_feature = "simd128",
        target_arch = "aarch64",
    )))]
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub(crate) unsafe fn find_structural_bits(
        input: &[u8],
        structural_indexes: &mut Vec<u32>,
    ) -> std::result::Result<(), ErrorType> {
        // This is a nasty hack, we don't have a chunked implementation for native rust
        // so we validate UTF8 ahead of time
        match core::str::from_utf8(input) {
            Ok(_) => (),
            Err(_) => return Err(ErrorType::InvalidUtf8),
        };
        #[cfg(not(feature = "portable"))]
        unsafe {
            Self::_find_structural_bits::<impls::native::SimdInput>(input, structural_indexes)
        }
    }

    #[cfg(all(feature = "portable", not(feature = "runtime-detection")))]
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub(crate) unsafe fn find_structural_bits(
        input: &[u8],
        structural_indexes: &mut Vec<u32>,
    ) -> std::result::Result<(), ErrorType> {
        unsafe {
            Self::_find_structural_bits::<impls::portable::SimdInput>(input, structural_indexes)
        }
    }

    #[cfg(all(
        target_feature = "avx2",
        not(feature = "portable"),
        not(feature = "runtime-detection"),
    ))]
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub(crate) unsafe fn find_structural_bits(
        input: &[u8],
        structural_indexes: &mut Vec<u32>,
    ) -> std::result::Result<(), ErrorType> {
        unsafe { Self::_find_structural_bits::<impls::avx2::SimdInput>(input, structural_indexes) }
    }

    #[cfg(all(
        target_feature = "sse4.2",
        not(target_feature = "avx2"),
        not(feature = "runtime-detection"),
        not(feature = "portable"),
    ))]
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub(crate) unsafe fn find_structural_bits(
        input: &[u8],
        structural_indexes: &mut Vec<u32>,
    ) -> std::result::Result<(), ErrorType> {
        unsafe { Self::_find_structural_bits::<impls::sse42::SimdInput>(input, structural_indexes) }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "portable")))]
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub(crate) unsafe fn find_structural_bits(
        input: &[u8],
        structural_indexes: &mut Vec<u32>,
    ) -> std::result::Result<(), ErrorType> {
        unsafe { Self::_find_structural_bits::<impls::neon::SimdInput>(input, structural_indexes) }
    }

    #[cfg(all(target_feature = "simd128", not(feature = "portable")))]
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub(crate) unsafe fn find_structural_bits(
        input: &[u8],
        structural_indexes: &mut Vec<u32>,
    ) -> std::result::Result<(), ErrorType> {
        unsafe {
            Self::_find_structural_bits::<impls::simd128::SimdInput>(input, structural_indexes)
        }
    }
}

impl<'de> Deserializer<'de> {
    /// Extracts the tape from the Deserializer
    #[must_use]
    pub fn into_tape(self) -> Tape<'de> {
        Tape(self.tape)
    }

    /// Gives a `Value` view of the tape in the Deserializer
    #[must_use]
    pub fn as_value(&self) -> Value<'_, 'de> {
        // Skip initial zero
        Value(&self.tape)
    }

    /// Resets the Deserializer tape index to 0
    pub fn restart(&mut self) {
        // Skip initial zero
        self.idx = 0;
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn error(error: ErrorType) -> Error {
        Error::new(0, None, error)
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn error_c(idx: usize, c: char, error: ErrorType) -> Error {
        Error::new(idx, Some(c), error)
    }

    /// Creates a serializer from a mutable slice of bytes
    ///
    /// # Errors
    ///
    /// Will return `Err` if `s` is invalid JSON.
    pub fn from_slice(input: &'de mut [u8]) -> Result<Self> {
        let len = input.len();

        let mut buffer = Buffers::new(len);

        Self::from_slice_with_buffers(input, &mut buffer)
    }

    /// Fills the tape without creating a serializer, this function poses
    /// lifetime chalanges and can be frustrating, howver when it is
    /// usable it allows a allocation free (armotized) parsing of JSON
    ///
    /// # Errors
    ///
    /// Will return `Err` if `input` is invalid JSON.
    #[allow(clippy::uninit_vec)]
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn fill_tape(
        input: &'de mut [u8],
        buffer: &mut Buffers,
        tape: &mut Vec<Node<'de>>,
    ) -> Result<()> {
        const LOTS_OF_ZOERS: [u8; SIMDINPUT_LENGTH] = [0; SIMDINPUT_LENGTH];
        let len = input.len();
        let simd_safe_len = len + SIMDINPUT_LENGTH;

        if len > u32::MAX as usize {
            return Err(Self::error(ErrorType::InputTooLarge));
        }

        buffer.string_buffer.clear();
        buffer.string_buffer.reserve(len + SIMDJSON_PADDING);

        unsafe {
            buffer.string_buffer.set_len(len + SIMDJSON_PADDING);
        };

        let input_buffer = &mut buffer.input_buffer;
        if input_buffer.capacity() < simd_safe_len {
            *input_buffer = AlignedBuf::with_capacity(simd_safe_len);
        }

        unsafe {
            input_buffer
                .as_mut_ptr()
                .copy_from_nonoverlapping(input.as_ptr(), len);

            // initialize all remaining bytes
            // this also ensures we have a 0 to terminate the buffer
            input_buffer
                .as_mut_ptr()
                .add(len)
                .copy_from_nonoverlapping(LOTS_OF_ZOERS.as_ptr(), SIMDINPUT_LENGTH);

            // safety: all bytes are initialized
            input_buffer.set_len(simd_safe_len);

            Self::find_structural_bits(input, &mut buffer.structural_indexes)
                .map_err(Error::generic)?;
        };

        Self::build_tape(
            input,
            input_buffer,
            &mut buffer.string_buffer,
            &buffer.structural_indexes,
            &mut buffer.stage2_stack,
            tape,
        )
    }

    /// Creates a serializer from a mutable slice of bytes using a temporary
    /// buffer for strings for them to be copied in and out if needed
    ///
    /// # Errors
    ///
    /// Will return `Err` if `s` is invalid JSON.
    pub fn from_slice_with_buffers(input: &'de mut [u8], buffer: &mut Buffers) -> Result<Self> {
        let mut tape: Vec<Node<'de>> = Vec::with_capacity(buffer.structural_indexes.len());

        Self::fill_tape(input, buffer, &mut tape)?;

        Ok(Self { tape, idx: 0 })
    }

    #[cfg(feature = "serde_impl")]
    #[cfg_attr(not(feature = "no-inline"), inline)]
    fn skip(&mut self) {
        self.idx += 1;
    }

    /// Same as `next()` but we pull out the check so we don't need to
    /// stry every time. Use this only if you know the next element exists!
    ///
    /// # Safety
    ///
    /// This function is not safe to use, it is meant for internal use
    /// where it's know the tape isn't finished.
    #[cfg_attr(not(feature = "no-inline"), inline)]
    pub unsafe fn next_(&mut self) -> Node<'de> {
        let r = *unsafe { self.tape.get_kinda_unchecked(self.idx) };
        self.idx += 1;
        r
    }

    #[cfg_attr(not(feature = "no-inline"), inline)]
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) unsafe fn _find_structural_bits<S: Stage1Parse>(
        input: &[u8],
        structural_indexes: &mut Vec<u32>,
    ) -> std::result::Result<(), ErrorType> {
        let len = input.len();
        // 8 is a heuristic number to estimate it turns out a rate of 1/8 structural characters
        // leads almost never to relocations.
        structural_indexes.clear();
        structural_indexes.reserve(len / 8);

        let mut utf8_validator = unsafe { S::Utf8Validator::new() };

        // we have padded the input out to 64 byte multiple with the remainder being
        // zeros

        // persistent state across loop
        // does the last iteration end with an odd-length sequence of backslashes?
        // either 0 or 1, but a 64-bit value
        let mut prev_iter_ends_odd_backslash: u64 = 0;
        // does the previous iteration end inside a double-quote pair?
        let mut prev_iter_inside_quote: u64 = 0;
        // either all zeros or all ones
        // does the previous iteration end on something that is a predecessor of a
        // pseudo-structural character - i.e. whitespace or a structural character
        // effectively the very first char is considered to follow "whitespace" for
        // the
        // purposes of pseudo-structural character detection so we initialize to 1
        let mut prev_iter_ends_pseudo_pred: u64 = 1;

        // structurals are persistent state across loop as we flatten them on the
        // subsequent iteration into our array pointed to be base_ptr.
        // This is harmless on the first iteration as structurals==0
        // and is done for performance reasons; we can hide some of the latency of the
        // expensive carryless multiply in the previous step with this work
        let mut structurals: u64 = 0;

        let lenminus64: usize = if len < 64 { 0 } else { len - 64 };
        let mut idx: usize = 0;
        let mut error_mask: u64 = 0; // for unescaped characters within strings (ASCII code points < 0x20)

        while idx < lenminus64 {
            /*
            #ifndef _MSC_VER
              __builtin_prefetch(buf + idx + 128);
            #endif
             */
            let chunk = unsafe { input.get_kinda_unchecked(idx..idx + 64) };
            unsafe { utf8_validator.update_from_chunks(chunk) };

            let input = unsafe { S::new(chunk) };
            // detect odd sequences of backslashes
            let odd_ends: u64 =
                input.find_odd_backslash_sequences(&mut prev_iter_ends_odd_backslash);

            // detect insides of quote pairs ("quote_mask") and also our quote_bits
            // themselves
            let mut quote_bits: u64 = 0;
            let quote_mask: u64 = input.find_quote_mask_and_bits(
                odd_ends,
                &mut prev_iter_inside_quote,
                &mut quote_bits,
                &mut error_mask,
            );

            // take the previous iterations structural bits, not our current iteration,
            // and flatten
            unsafe { S::flatten_bits(structural_indexes, idx as u32, structurals) };

            let mut whitespace: u64 = 0;
            unsafe { input.find_whitespace_and_structurals(&mut whitespace, &mut structurals) };

            // fixup structurals to reflect quotes and add pseudo-structural characters
            structurals = S::finalize_structurals(
                structurals,
                whitespace,
                quote_mask,
                quote_bits,
                &mut prev_iter_ends_pseudo_pred,
            );
            idx += SIMDINPUT_LENGTH;
        }

        // we use a giant copy-paste which is ugly.
        // but otherwise the string needs to be properly padded or else we
        // risk invalidating the UTF-8 checks.
        if idx < len {
            let mut tmpbuf: [u8; SIMDINPUT_LENGTH] = [0x20; SIMDINPUT_LENGTH];
            unsafe {
                tmpbuf
                    .as_mut_ptr()
                    .copy_from(input.as_ptr().add(idx), len - idx);
            };
            unsafe { utf8_validator.update_from_chunks(&tmpbuf) };

            let input = unsafe { S::new(&tmpbuf) };

            // detect odd sequences of backslashes
            let odd_ends: u64 =
                input.find_odd_backslash_sequences(&mut prev_iter_ends_odd_backslash);

            // detect insides of quote pairs ("quote_mask") and also our quote_bits
            // themselves
            let mut quote_bits: u64 = 0;
            let quote_mask: u64 = input.find_quote_mask_and_bits(
                odd_ends,
                &mut prev_iter_inside_quote,
                &mut quote_bits,
                &mut error_mask,
            );

            // take the previous iterations structural bits, not our current iteration,
            // and flatten
            unsafe { S::flatten_bits(structural_indexes, idx as u32, structurals) };

            let mut whitespace: u64 = 0;
            unsafe { input.find_whitespace_and_structurals(&mut whitespace, &mut structurals) };

            // fixup structurals to reflect quotes and add pseudo-structural characters
            structurals = S::finalize_structurals(
                structurals,
                whitespace,
                quote_mask,
                quote_bits,
                &mut prev_iter_ends_pseudo_pred,
            );
            idx += SIMDINPUT_LENGTH;
        }
        // This test isn't in upstream, for some reason the error mask is et for then.
        if prev_iter_inside_quote != 0 {
            return Err(ErrorType::Syntax);
        }
        // finally, flatten out the remaining structurals from the last iteration
        unsafe { S::flatten_bits(structural_indexes, idx as u32, structurals) };

        // a valid JSON file cannot have zero structural indexes - we should have
        // found something (note that we compare to 1 as we always add the root!)
        if structural_indexes.is_empty() {
            return Err(ErrorType::Eof);
        }

        if error_mask != 0 {
            return Err(ErrorType::Syntax);
        }

        if unsafe { utf8_validator.finalize(None).is_err() } {
            Err(ErrorType::InvalidUtf8)
        } else {
            Ok(())
        }
    }
}

/// SIMD aligned buffer
struct AlignedBuf {
    layout: Layout,
    capacity: usize,
    len: usize,
    inner: NonNull<u8>,
}
// We use allow Sync + Send here since we know u8 is sync and send
// we never reallocate or grow this buffer only allocate it in
// create then deallocate it in drop.
//
// An example of this can be found [in the official rust docs](https://doc.rust-lang.org/nomicon/vec/vec-raw.html).

unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}
impl AlignedBuf {
    /// Creates a new buffer that is  aligned with the simd register size
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let Ok(layout) = Layout::from_size_align(capacity, SIMDJSON_PADDING) else {
            Self::capacity_overflow()
        };
        if mem::size_of::<usize>() < 8 && capacity > isize::MAX as usize {
            Self::capacity_overflow()
        }
        unsafe {
            let Some(inner) = NonNull::new(alloc(layout)) else {
                handle_alloc_error(layout)
            };
            Self {
                layout,
                capacity,
                len: 0,
                inner,
            }
        }
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.inner.as_ptr()
    }

    fn capacity_overflow() -> ! {
        panic!("capacity overflow");
    }
    fn capacity(&self) -> usize {
        self.capacity
    }
    unsafe fn set_len(&mut self, n: usize) {
        assert!(
            n <= self.capacity,
            "New size ({}) can not be larger then capacity ({}).",
            n,
            self.capacity
        );
        self.len = n;
    }
}
impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.inner.as_ptr(), self.layout);
        }
    }
}

impl Deref for AlignedBuf {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.inner.as_ptr(), self.len) }
    }
}

impl DerefMut for AlignedBuf {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.inner.as_ptr(), self.len) }
    }
}
