#![no_std]
//! Minimal safe wrapper around zstd-sys.
//!
//! This crates provides a minimal translation of the [zstd-sys] methods.
//! For a more comfortable high-level library, see the [zstd] crate.
//!
//! [zstd-sys]: https://crates.io/crates/zstd-sys
//! [zstd]: https://crates.io/crates/zstd
//!
//! Most of the functions here map 1-for-1 to a function from
//! [the C zstd library][zstd-c] mentioned in their descriptions.
//! Check the [source documentation][doc] for more information on their
//! behaviour.
//!
//! [doc]: https://facebook.github.io/zstd/zstd_manual.html
//! [zstd-c]: https://facebook.github.io/zstd/
//!
//! Features denoted as experimental in the C library are hidden behind an
//! `experimental` feature.
#![cfg_attr(feature = "doc-cfg", feature(doc_cfg))]

// TODO: Use alloc feature instead to implement stuff for Vec
// TODO: What about Cursor?
#[cfg(feature = "std")]
extern crate std;

#[cfg(test)]
mod tests;

#[cfg(feature = "seekable")]
pub mod seekable;

// Re-export zstd-sys
pub use zstd_sys;

/// How to compress data.
pub use zstd_sys::ZSTD_strategy as Strategy;

/// Frame progression state.
#[cfg(feature = "experimental")]
#[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
pub use zstd_sys::ZSTD_frameProgression as FrameProgression;

/// Reset directive.
// pub use zstd_sys::ZSTD_ResetDirective as ResetDirective;
use core::ffi::{c_char, c_int, c_ulonglong, c_void};

use core::marker::PhantomData;
use core::num::{NonZeroU32, NonZeroU64};
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;
use core::str;

include!("constants.rs");

#[cfg(feature = "experimental")]
include!("constants_experimental.rs");

#[cfg(feature = "seekable")]
include!("constants_seekable.rs");

/// Represents the compression level used by zstd.
pub type CompressionLevel = i32;

/// Represents a possible error from the zstd library.
pub type ErrorCode = usize;

/// Wrapper result around most zstd functions.
///
/// Either a success code (usually number of bytes written), or an error code.
pub type SafeResult = Result<usize, ErrorCode>;

/// Indicates an error happened when parsing the frame content size.
///
/// The stream may be corrupted, or the given frame prefix was too small.
#[derive(Debug)]
pub struct ContentSizeError;

impl core::fmt::Display for ContentSizeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("Could not get content size")
    }
}

/// Returns true if code represents error.
fn is_error(code: usize) -> bool {
    // Safety: Just FFI
    unsafe { zstd_sys::ZSTD_isError(code) != 0 }
}

/// Parse the result code
///
/// Returns the number of bytes written if the code represents success,
/// or the error message code otherwise.
fn parse_code(code: usize) -> SafeResult {
    if !is_error(code) {
        Ok(code)
    } else {
        Err(code)
    }
}

/// Parse a content size value.
///
/// zstd uses 2 special content size values to indicate either unknown size or parsing error.
fn parse_content_size(
    content_size: u64,
) -> Result<Option<u64>, ContentSizeError> {
    match content_size {
        CONTENTSIZE_ERROR => Err(ContentSizeError),
        CONTENTSIZE_UNKNOWN => Ok(None),
        other => Ok(Some(other)),
    }
}

fn ptr_void(src: &[u8]) -> *const c_void {
    src.as_ptr() as *const c_void
}

fn ptr_mut_void(dst: &mut (impl WriteBuf + ?Sized)) -> *mut c_void {
    dst.as_mut_ptr() as *mut c_void
}

/// Returns the ZSTD version.
///
/// Returns `major * 10_000 + minor * 100 + patch`.
/// So 1.5.3 would be returned as `10_503`.
pub fn version_number() -> u32 {
    // Safety: Just FFI
    unsafe { zstd_sys::ZSTD_versionNumber() as u32 }
}

/// Returns a string representation of the ZSTD version.
///
/// For example "1.5.3".
pub fn version_string() -> &'static str {
    // Safety: Assumes `ZSTD_versionString` returns a valid utf8 string.
    unsafe { c_char_to_str(zstd_sys::ZSTD_versionString()) }
}

/// Returns the minimum (fastest) compression level supported.
///
/// This is likely going to be a _very_ large negative number.
pub fn min_c_level() -> CompressionLevel {
    // Safety: Just FFI
    unsafe { zstd_sys::ZSTD_minCLevel() as CompressionLevel }
}

/// Returns the maximum (slowest) compression level supported.
pub fn max_c_level() -> CompressionLevel {
    // Safety: Just FFI
    unsafe { zstd_sys::ZSTD_maxCLevel() as CompressionLevel }
}

/// Wraps the `ZSTD_compress` function.
///
/// This will try to compress `src` entirely and write the result to `dst`, returning the number of
/// bytes written. If `dst` is too small to hold the compressed content, an error will be returned.
///
/// For streaming operations that don't require to store the entire input/output in memory, see
/// `compress_stream`.
pub fn compress<C: WriteBuf + ?Sized>(
    dst: &mut C,
    src: &[u8],
    compression_level: CompressionLevel,
) -> SafeResult {
    // Safety: ZSTD_compress indeed returns how many bytes have been written.
    unsafe {
        dst.write_from(|buffer, capacity| {
            parse_code(zstd_sys::ZSTD_compress(
                buffer,
                capacity,
                ptr_void(src),
                src.len(),
                compression_level,
            ))
        })
    }
}

/// Wraps the `ZSTD_decompress` function.
///
/// This is a one-step decompression (not streaming).
///
/// You will need to make sure `dst` is large enough to store all the decompressed content, or an
/// error will be returned.
///
/// If decompression was a success, the number of bytes written will be returned.
pub fn decompress<C: WriteBuf + ?Sized>(
    dst: &mut C,
    src: &[u8],
) -> SafeResult {
    // Safety: ZSTD_decompress indeed returns how many bytes have been written.
    unsafe {
        dst.write_from(|buffer, capacity| {
            parse_code(zstd_sys::ZSTD_decompress(
                buffer,
                capacity,
                ptr_void(src),
                src.len(),
            ))
        })
    }
}

/// Wraps the `ZSTD_getDecompressedSize` function.
///
/// Returns `None` if the size could not be found, or if the content is actually empty.
#[deprecated(note = "Use ZSTD_getFrameContentSize instead")]
pub fn get_decompressed_size(src: &[u8]) -> Option<NonZeroU64> {
    // Safety: Just FFI
    NonZeroU64::new(unsafe {
        zstd_sys::ZSTD_getDecompressedSize(ptr_void(src), src.len()) as u64
    })
}

/// Maximum compressed size in worst case single-pass scenario
pub fn compress_bound(src_size: usize) -> usize {
    // Safety: Just FFI
    unsafe { zstd_sys::ZSTD_compressBound(src_size) }
}

/// Compression context
///
/// It is recommended to allocate a single context per thread and re-use it
/// for many compression operations.
pub struct CCtx<'a>(NonNull<zstd_sys::ZSTD_CCtx>, PhantomData<&'a ()>);

impl Default for CCtx<'_> {
    fn default() -> Self {
        CCtx::create()
    }
}

impl<'a> CCtx<'a> {
    /// Tries to create a new context.
    ///
    /// Returns `None` if zstd returns a NULL pointer - may happen if allocation fails.
    pub fn try_create() -> Option<Self> {
        // Safety: Just FFI
        Some(CCtx(
            NonNull::new(unsafe { zstd_sys::ZSTD_createCCtx() })?,
            PhantomData,
        ))
    }

    /// Wrap `ZSTD_createCCtx`
    ///
    /// # Panics
    ///
    /// If zstd returns a NULL pointer.
    pub fn create() -> Self {
        Self::try_create()
            .expect("zstd returned null pointer when creating new context")
    }

    /// Wraps the `ZSTD_compressCCtx()` function
    pub fn compress<C: WriteBuf + ?Sized>(
        &mut self,
        dst: &mut C,
        src: &[u8],
        compression_level: CompressionLevel,
    ) -> SafeResult {
        // Safety: ZSTD_compressCCtx returns how many bytes were written.
        unsafe {
            dst.write_from(|buffer, capacity| {
                parse_code(zstd_sys::ZSTD_compressCCtx(
                    self.0.as_ptr(),
                    buffer,
                    capacity,
                    ptr_void(src),
                    src.len(),
                    compression_level,
                ))
            })
        }
    }

    /// Wraps the `ZSTD_compress2()` function.
    pub fn compress2<C: WriteBuf + ?Sized>(
        &mut self,
        dst: &mut C,
        src: &[u8],
    ) -> SafeResult {
        // Safety: ZSTD_compress2 returns how many bytes were written.
        unsafe {
            dst.write_from(|buffer, capacity| {
                parse_code(zstd_sys::ZSTD_compress2(
                    self.0.as_ptr(),
                    buffer,
                    capacity,
                    ptr_void(src),
                    src.len(),
                ))
            })
        }
    }

    /// Wraps the `ZSTD_compress_usingDict()` function.
    pub fn compress_using_dict<C: WriteBuf + ?Sized>(
        &mut self,
        dst: &mut C,
        src: &[u8],
        dict: &[u8],
        compression_level: CompressionLevel,
    ) -> SafeResult {
        // Safety: ZSTD_compress_usingDict returns how many bytes were written.
        unsafe {
            dst.write_from(|buffer, capacity| {
                parse_code(zstd_sys::ZSTD_compress_usingDict(
                    self.0.as_ptr(),
                    buffer,
                    capacity,
                    ptr_void(src),
                    src.len(),
                    ptr_void(dict),
                    dict.len(),
                    compression_level,
                ))
            })
        }
    }

    /// Wraps the `ZSTD_compress_usingCDict()` function.
    pub fn compress_using_cdict<C: WriteBuf + ?Sized>(
        &mut self,
        dst: &mut C,
        src: &[u8],
        cdict: &CDict<'_>,
    ) -> SafeResult {
        // Safety: ZSTD_compress_usingCDict returns how many bytes were written.
        unsafe {
            dst.write_from(|buffer, capacity| {
                parse_code(zstd_sys::ZSTD_compress_usingCDict(
                    self.0.as_ptr(),
                    buffer,
                    capacity,
                    ptr_void(src),
                    src.len(),
                    cdict.0.as_ptr(),
                ))
            })
        }
    }

    /// Initializes the context with the given compression level.
    ///
    /// This is equivalent to running:
    /// * `reset()`
    /// * `set_parameter(CompressionLevel, compression_level)`
    pub fn init(&mut self, compression_level: CompressionLevel) -> SafeResult {
        // Safety: Just FFI
        let code = unsafe {
            zstd_sys::ZSTD_initCStream(self.0.as_ptr(), compression_level)
        };
        parse_code(code)
    }

    /// Wraps the `ZSTD_initCStream_srcSize()` function.
    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    #[deprecated]
    pub fn init_src_size(
        &mut self,
        compression_level: CompressionLevel,
        pledged_src_size: u64,
    ) -> SafeResult {
        // Safety: Just FFI
        let code = unsafe {
            zstd_sys::ZSTD_initCStream_srcSize(
                self.0.as_ptr(),
                compression_level as c_int,
                pledged_src_size as c_ulonglong,
            )
        };
        parse_code(code)
    }

    /// Wraps the `ZSTD_initCStream_usingDict()` function.
    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    #[deprecated]
    pub fn init_using_dict(
        &mut self,
        dict: &[u8],
        compression_level: CompressionLevel,
    ) -> SafeResult {
        // Safety: Just FFI
        let code = unsafe {
            zstd_sys::ZSTD_initCStream_usingDict(
                self.0.as_ptr(),
                ptr_void(dict),
                dict.len(),
                compression_level,
            )
        };
        parse_code(code)
    }

    /// Wraps the `ZSTD_initCStream_usingCDict()` function.
    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    #[deprecated]
    pub fn init_using_cdict<'b>(&mut self, cdict: &CDict<'b>) -> SafeResult
    where
        'b: 'a, // Dictionary outlives the stream.
    {
        // Safety: Just FFI
        let code = unsafe {
            zstd_sys::ZSTD_initCStream_usingCDict(
                self.0.as_ptr(),
                cdict.0.as_ptr(),
            )
        };
        parse_code(code)
    }

    /// Tries to load a dictionary.
    ///
    /// The dictionary content will be copied internally and does not need to be kept alive after
    /// calling this function.
    ///
    /// If you need to use the same dictionary for multiple contexts, it may be more efficient to
    /// create a `CDict` first, then loads that.
    ///
    /// The dictionary will apply to all compressed frames, until a new dictionary is set.
    pub fn load_dictionary(&mut self, dict: &[u8]) -> SafeResult {
        // Safety: Just FFI
        parse_code(unsafe {
            zstd_sys::ZSTD_CCtx_loadDictionary(
                self.0.as_ptr(),
                ptr_void(dict),
                dict.len(),
            )
        })
    }

    /// Wraps the `ZSTD_CCtx_refCDict()` function.
    ///
    /// Dictionary must outlive the context.
    pub fn ref_cdict<'b>(&mut self, cdict: &CDict<'b>) -> SafeResult
    where
        'b: 'a,
    {
        // Safety: Just FFI
        parse_code(unsafe {
            zstd_sys::ZSTD_CCtx_refCDict(self.0.as_ptr(), cdict.0.as_ptr())
        })
    }

    /// Return to "no-dictionary" mode.
    ///
    /// This will disable any dictionary/prefix previously registered for future frames.
    pub fn disable_dictionary(&mut self) -> SafeResult {
        // Safety: Just FFI
        parse_code(unsafe {
            zstd_sys::ZSTD_CCtx_loadDictionary(
                self.0.as_ptr(),
                core::ptr::null(),
                0,
            )
        })
    }

    /// Use some prefix as single-use dictionary for the next compressed frame.
    ///
    /// Just like a dictionary, decompression will need to be given the same prefix.
    ///
    /// This is best used if the "prefix" looks like the data to be compressed.
    pub fn ref_prefix<'b>(&mut self, prefix: &'b [u8]) -> SafeResult
    where
        'b: 'a,
    {
        // Safety: Just FFI
        parse_code(unsafe {
            zstd_sys::ZSTD_CCtx_refPrefix(
                self.0.as_ptr(),
                ptr_void(prefix),
                prefix.len(),
            )
        })
    }

    /// Performs a step of a streaming compression operation.
    ///
    /// This will read some data from `input` and/or write some data to `output`.
    ///
    /// # Returns
    ///
    /// A hint for the "ideal" amount of input data to provide in the next call.
    ///
    /// This hint is only for performance purposes.
    ///
    /// Wraps the `ZSTD_compressStream()` function.
    pub fn compress_stream<C: WriteBuf + ?Sized>(
        &mut self,
        output: &mut OutBuffer<'_, C>,
        input: &mut InBuffer<'_>,
    ) -> SafeResult {
        let mut output = output.wrap();
        let mut input = input.wrap();
        // Safety: Just FFI
        let code = unsafe {
            zstd_sys::ZSTD_compressStream(
                self.0.as_ptr(),
                ptr_mut(&mut output),
                ptr_mut(&mut input),
            )
        };
        parse_code(code)
    }

    /// Performs a step of a streaming compression operation.
    ///
    /// This will read some data from `input` and/or write some data to `output`.
    ///
    /// The `end_op` directive can be used to specify what to do after: nothing special, flush
    /// internal buffers, or end the frame.
    ///
    /// # Returns
    ///
    /// An lower bound for the amount of data that still needs to be flushed out.
    ///
    /// This is useful when flushing or ending the frame: you need to keep calling this function
    /// until it returns 0.
    ///
    /// Wraps the `ZSTD_compressStream2()` function.
    pub fn compress_stream2<C: WriteBuf + ?Sized>(
        &mut self,
        output: &mut OutBuffer<'_, C>,
        input: &mut InBuffer<'_>,
        end_op: zstd_sys::ZSTD_EndDirective,
    ) -> SafeResult {
        let mut output = output.wrap();
        let mut input = input.wrap();
        // Safety: Just FFI
        parse_code(unsafe {
            zstd_sys::ZSTD_compressStream2(
                self.0.as_ptr(),
                ptr_mut(&mut output),
                ptr_mut(&mut input),
                end_op,
            )
        })
    }

    /// Flush any intermediate buffer.
    ///
    /// To fully flush, you should keep calling this function until it returns `Ok(0)`.
    ///
    /// Wraps the `ZSTD_flushStream()` function.
    pub fn flush_stream<C: WriteBuf + ?Sized>(
        &mut self,
        output: &mut OutBuffer<'_, C>,
    ) -> SafeResult {
        let mut output = output.wrap();
        // Safety: Just FFI
        let code = unsafe {
            zstd_sys::ZSTD_flushStream(self.0.as_ptr(), ptr_mut(&mut output))
        };
        parse_code(code)
    }

    /// Ends the stream.
    ///
    /// You should keep calling this function until it returns `Ok(0)`.
    ///
    /// Wraps the `ZSTD_endStream()` function.
    pub fn end_stream<C: WriteBuf + ?Sized>(
        &mut self,
        output: &mut OutBuffer<'_, C>,
    ) -> SafeResult {
        let mut output = output.wrap();
        // Safety: Just FFI
        let code = unsafe {
            zstd_sys::ZSTD_endStream(self.0.as_ptr(), ptr_mut(&mut output))
        };
        parse_code(code)
    }

    /// Returns the size currently used by this context.
    ///
    /// This may change over time.
    pub fn sizeof(&self) -> usize {
        // Safety: Just FFI
        unsafe { zstd_sys::ZSTD_sizeof_CCtx(self.0.as_ptr()) }
    }

    /// Resets the state of the context.
    ///
    /// Depending on the reset mode, it can reset the session, the parameters, or both.
    ///
    /// Wraps the `ZSTD_CCtx_reset()` function.
    pub fn reset(&mut self, reset: ResetDirective) -> SafeResult {
        // Safety: Just FFI
        parse_code(unsafe {
            zstd_sys::ZSTD_CCtx_reset(self.0.as_ptr(), reset.as_sys())
        })
    }

    /// Sets a compression parameter.
    ///
    /// Some of these parameters need to be set during de-compression as well.
    pub fn set_parameter(&mut self, param: CParameter) -> SafeResult {
        // TODO: Until bindgen properly generates a binding for this, we'll need to do it here.

        #[cfg(feature = "experimental")]
        use zstd_sys::ZSTD_cParameter::{
            ZSTD_c_experimentalParam1 as ZSTD_c_rsyncable,
            ZSTD_c_experimentalParam10 as ZSTD_c_stableOutBuffer,
            ZSTD_c_experimentalParam11 as ZSTD_c_blockDelimiters,
            ZSTD_c_experimentalParam12 as ZSTD_c_validateSequences,
            ZSTD_c_experimentalParam13 as ZSTD_c_useBlockSplitter,
            ZSTD_c_experimentalParam14 as ZSTD_c_useRowMatchFinder,
            ZSTD_c_experimentalParam15 as ZSTD_c_deterministicRefPrefix,
            ZSTD_c_experimentalParam16 as ZSTD_c_prefetchCDictTables,
            ZSTD_c_experimentalParam17 as ZSTD_c_enableSeqProducerFallback,
            ZSTD_c_experimentalParam18 as ZSTD_c_maxBlockSize,
            ZSTD_c_experimentalParam19 as ZSTD_c_searchForExternalRepcodes,
            ZSTD_c_experimentalParam2 as ZSTD_c_format,
            ZSTD_c_experimentalParam3 as ZSTD_c_forceMaxWindow,
            ZSTD_c_experimentalParam4 as ZSTD_c_forceAttachDict,
            ZSTD_c_experimentalParam5 as ZSTD_c_literalCompressionMode,
            ZSTD_c_experimentalParam7 as ZSTD_c_srcSizeHint,
            ZSTD_c_experimentalParam8 as ZSTD_c_enableDedicatedDictSearch,
            ZSTD_c_experimentalParam9 as ZSTD_c_stableInBuffer,
        };

        use zstd_sys::ZSTD_cParameter::*;
        use CParameter::*;

        let (param, value) = match param {
            #[cfg(feature = "experimental")]
            RSyncable(rsyncable) => (ZSTD_c_rsyncable, rsyncable as c_int),
            #[cfg(feature = "experimental")]
            Format(format) => (ZSTD_c_format, format as c_int),
            #[cfg(feature = "experimental")]
            ForceMaxWindow(force) => (ZSTD_c_forceMaxWindow, force as c_int),
            #[cfg(feature = "experimental")]
            ForceAttachDict(force) => (ZSTD_c_forceAttachDict, force as c_int),
            #[cfg(feature = "experimental")]
            LiteralCompressionMode(mode) => {
                (ZSTD_c_literalCompressionMode, mode as c_int)
            }
            #[cfg(feature = "experimental")]
            SrcSizeHint(value) => (ZSTD_c_srcSizeHint, value as c_int),
            #[cfg(feature = "experimental")]
            EnableDedicatedDictSearch(enable) => {
                (ZSTD_c_enableDedicatedDictSearch, enable as c_int)
            }
            #[cfg(feature = "experimental")]
            StableInBuffer(stable) => (ZSTD_c_stableInBuffer, stable as c_int),
            #[cfg(feature = "experimental")]
            StableOutBuffer(stable) => {
                (ZSTD_c_stableOutBuffer, stable as c_int)
            }
            #[cfg(feature = "experimental")]
            BlockDelimiters(value) => (ZSTD_c_blockDelimiters, value as c_int),
            #[cfg(feature = "experimental")]
            ValidateSequences(validate) => {
                (ZSTD_c_validateSequences, validate as c_int)
            }
            #[cfg(feature = "experimental")]
            UseBlockSplitter(split) => {
                (ZSTD_c_useBlockSplitter, split as c_int)
            }
            #[cfg(feature = "experimental")]
            UseRowMatchFinder(mode) => {
                (ZSTD_c_useRowMatchFinder, mode as c_int)
            }
            #[cfg(feature = "experimental")]
            DeterministicRefPrefix(deterministic) => {
                (ZSTD_c_deterministicRefPrefix, deterministic as c_int)
            }
            #[cfg(feature = "experimental")]
            PrefetchCDictTables(prefetch) => {
                (ZSTD_c_prefetchCDictTables, prefetch as c_int)
            }
            #[cfg(feature = "experimental")]
            EnableSeqProducerFallback(enable) => {
                (ZSTD_c_enableSeqProducerFallback, enable as c_int)
            }
            #[cfg(feature = "experimental")]
            MaxBlockSize(value) => (ZSTD_c_maxBlockSize, value as c_int),
            #[cfg(feature = "experimental")]
            SearchForExternalRepcodes(value) => {
                (ZSTD_c_searchForExternalRepcodes, value as c_int)
            }
            TargetCBlockSize(value) => {
                (ZSTD_c_targetCBlockSize, value as c_int)
            }
            CompressionLevel(level) => (ZSTD_c_compressionLevel, level),
            WindowLog(value) => (ZSTD_c_windowLog, value as c_int),
            HashLog(value) => (ZSTD_c_hashLog, value as c_int),
            ChainLog(value) => (ZSTD_c_chainLog, value as c_int),
            SearchLog(value) => (ZSTD_c_searchLog, value as c_int),
            MinMatch(value) => (ZSTD_c_minMatch, value as c_int),
            TargetLength(value) => (ZSTD_c_targetLength, value as c_int),
            Strategy(strategy) => (ZSTD_c_strategy, strategy as c_int),
            EnableLongDistanceMatching(flag) => {
                (ZSTD_c_enableLongDistanceMatching, flag as c_int)
            }
            LdmHashLog(value) => (ZSTD_c_ldmHashLog, value as c_int),
            LdmMinMatch(value) => (ZSTD_c_ldmMinMatch, value as c_int),
            LdmBucketSizeLog(value) => {
                (ZSTD_c_ldmBucketSizeLog, value as c_int)
            }
            LdmHashRateLog(value) => (ZSTD_c_ldmHashRateLog, value as c_int),
            ContentSizeFlag(flag) => (ZSTD_c_contentSizeFlag, flag as c_int),
            ChecksumFlag(flag) => (ZSTD_c_checksumFlag, flag as c_int),
            DictIdFlag(flag) => (ZSTD_c_dictIDFlag, flag as c_int),

            NbWorkers(value) => (ZSTD_c_nbWorkers, value as c_int),

            JobSize(value) => (ZSTD_c_jobSize, value as c_int),

            OverlapSizeLog(value) => (ZSTD_c_overlapLog, value as c_int),
        };

        // Safety: Just FFI
        parse_code(unsafe {
            zstd_sys::ZSTD_CCtx_setParameter(self.0.as_ptr(), param, value)
        })
    }

    /// Guarantee that the input size will be this value.
    ///
    /// If given `None`, assumes the size is unknown.
    ///
    /// Unless explicitly disabled, this will cause the size to be written in the compressed frame
    /// header.
    ///
    /// If the actual data given to compress has a different size, an error will be returned.
    pub fn set_pledged_src_size(
        &mut self,
        pledged_src_size: Option<u64>,
    ) -> SafeResult {
        // Safety: Just FFI
        parse_code(unsafe {
            zstd_sys::ZSTD_CCtx_setPledgedSrcSize(
                self.0.as_ptr(),
                pledged_src_size.unwrap_or(CONTENTSIZE_UNKNOWN) as c_ulonglong,
            )
        })
    }

    /// Creates a copy of this context.
    ///
    /// This only works before any data has been compressed. An error will be
    /// returned otherwise.
    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    pub fn try_clone(
        &self,
        pledged_src_size: Option<u64>,
    ) -> Result<Self, ErrorCode> {
        // Safety: Just FFI
        let context = NonNull::new(unsafe { zstd_sys::ZSTD_createCCtx() })
            .ok_or(0usize)?;

        // Safety: Just FFI
        parse_code(unsafe {
            zstd_sys::ZSTD_copyCCtx(
                context.as_ptr(),
                self.0.as_ptr(),
                pledged_src_size.unwrap_or(CONTENTSIZE_UNKNOWN),
            )
        })?;

        Ok(CCtx(context, self.1))
    }

    /// Wraps the `ZSTD_getBlockSize()` function.
    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    pub fn get_block_size(&self) -> usize {
        // Safety: Just FFI
        unsafe { zstd_sys::ZSTD_getBlockSize(self.0.as_ptr()) }
    }

    /// Wraps the `ZSTD_compressBlock()` function.
    ///
    /// # Safety
    ///
    /// `src` becomes this context's history window, so it must stay allocated and unmodified until
    /// the next call to `compress_block` on this context (or until this context is dropped), as the
    /// following block is compressed against it.
    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    pub unsafe fn compress_block<C: WriteBuf + ?Sized>(
        &mut self,
        dst: &mut C,
        src: &[u8],
    ) -> SafeResult {
        // Safety: ZSTD_compressBlock returns the number of bytes written.
        unsafe {
            dst.write_from(|buffer, capacity| {
                parse_code(zstd_sys::ZSTD_compressBlock(
                    self.0.as_ptr(),
                    buffer,
                    capacity,
                    ptr_void(src),
                    src.len(),
                ))
            })
        }
    }

    /// Returns the recommended input buffer size.
    ///
    /// Using this size may result in minor performance boost.
    pub fn in_size() -> usize {
        // Safety: Just FFI
        unsafe { zstd_sys::ZSTD_CStreamInSize() }
    }

    /// Returns the recommended output buffer size.
    ///
    /// Using this may result in minor performance boost.
    pub fn out_size() -> usize {
        // Safety: Just FFI
        unsafe { zstd_sys::ZSTD_CStreamOutSize() }
    }

    /// Use a shared thread pool for this context.
    ///
    /// Thread pool must outlive the context.
    #[cfg(all(feature = "experimental", feature = "zstdmt"))]
    #[cfg_attr(
        feature = "doc-cfg",
        doc(cfg(all(feature = "experimental", feature = "zstdmt")))
    )]
    pub fn ref_thread_pool<'b>(&mut self, pool: &'b ThreadPool) -> SafeResult
    where
        'b: 'a,
    {
        parse_code(unsafe {
            zstd_sys::ZSTD_CCtx_refThreadPool(self.0.as_ptr(), pool.0.as_ptr())
        })
    }

    /// Return to using a private thread pool for this context.
    #[cfg(all(feature = "experimental", feature = "zstdmt"))]
    #[cfg_attr(
        feature = "doc-cfg",
        doc(cfg(all(feature = "experimental", feature = "zstdmt")))
    )]
    pub fn disable_thread_pool(&mut self) -> SafeResult {
        parse_code(unsafe {
            zstd_sys::ZSTD_CCtx_refThreadPool(
                self.0.as_ptr(),
                core::ptr::null_mut(),
            )
        })
    }

    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    pub fn get_frame_progression(&self) -> FrameProgression {
        // Safety: Just FFI
        unsafe { zstd_sys::ZSTD_getFrameProgression(self.0.as_ptr()) }
    }
}

impl<'a> Drop for CCtx<'a> {
    fn drop(&mut self) {
        // Safety: Just FFI
        unsafe {
            zstd_sys::ZSTD_freeCCtx(self.0.as_ptr());
        }
    }
}

unsafe impl Send for CCtx<'_> {}
// Non thread-safe methods already take `&mut self`, so it's fine to implement Sync here.
unsafe impl Sync for CCtx<'_> {}

unsafe fn c_char_to_str(text: *const c_char) -> &'static str {
    core::ffi::CStr::from_ptr(text)
        .to_str()
        .expect("bad error message from zstd")
}

/// Returns the error string associated with an error code.
pub fn get_error_name(code: usize) -> &'static str {
    unsafe {
        // Safety: assumes ZSTD returns a well-formed utf8 string.
        let name = zstd_sys::ZSTD_getErrorName(code);
        c_char_to_str(name)
    }
}

/// A Decompression Context.
///
/// The lifetime references the potential dictionary used for this context.
///
/// If no dictionary was used, it will most likely be `'static`.
///
/// Same as `DStream`.
pub struct DCtx<'a>(NonNull<zstd_sys::ZSTD_DCtx>, PhantomData<&'a ()>);

impl Default for DCtx<'_> {
    fn default() -> Self {
        DCtx::create()
    }
}

impl<'a> DCtx<'a> {
    /// Try to create a new decompression context.
    ///
    /// Returns `None` if the operation failed (for example, not enough memory).
    pub fn try_create() -> Option<Self> {
        Some(DCtx(
            NonNull::new(unsafe { zstd_sys::ZSTD_createDCtx() })?,
            PhantomData,
        ))
    }

    /// Creates a new decoding context.
    ///
    /// # Panics
    ///
    /// If the context creation fails.
    pub fn create() -> Self {
        Self::try_create()
            .expect("zstd returned null pointer when creating new context")
    }

    /// Fully decompress the given frame.
    ///
    /// This decompress an entire frame in-memory. If you can have enough memory to store both the
    /// input and output buffer, then it may be faster that streaming decompression.
    ///
    /// Wraps the `ZSTD_decompressDCtx()` function.
    pub fn decompress<C: WriteBuf + ?Sized>(
        &mut self,
        dst: &mut C,
        src: &[u8],
    ) -> SafeResult {
        unsafe {
            dst.write_from(|buffer, capacity| {
                parse_code(zstd_sys::ZSTD_decompressDCtx(
                    self.0.as_ptr(),
                    buffer,
                    capacity,
                    ptr_void(src),
                    src.len(),
                ))
            })
        }
    }

    /// Fully decompress the given frame using a dictionary.
    ///
    /// Dictionary must be identical to the one used during compression.
    ///
    /// If you plan on using the same dictionary multiple times, it is faster to create a `DDict`
    /// first and use `decompress_using_ddict`.
    ///
    /// Wraps `ZSTD_decompress_usingDict`
    pub fn decompress_using_dict<C: WriteBuf + ?Sized>(
        &mut self,
        dst: &mut C,
        src: &[u8],
        dict: &[u8],
    ) -> SafeResult {
        unsafe {
            dst.write_from(|buffer, capacity| {
                parse_code(zstd_sys::ZSTD_decompress_usingDict(
                    self.0.as_ptr(),
                    buffer,
                    capacity,
                    ptr_void(src),
                    src.len(),
                    ptr_void(dict),
                    dict.len(),
                ))
            })
        }
    }

    /// Fully decompress the given frame using a dictionary.
    ///
    /// Dictionary must be identical to the one used during compression.
    ///
    /// Wraps the `ZSTD_decompress_usingDDict()` function.
    pub fn decompress_using_ddict<C: WriteBuf + ?Sized>(
        &mut self,
        dst: &mut C,
        src: &[u8],
        ddict: &DDict<'_>,
    ) -> SafeResult {
        unsafe {
            dst.write_from(|buffer, capacity| {
                parse_code(zstd_sys::ZSTD_decompress_usingDDict(
                    self.0.as_ptr(),
                    buffer,
                    capacity,
                    ptr_void(src),
                    src.len(),
                    ddict.0.as_ptr(),
                ))
            })
        }
    }

    /// Initializes an existing `DStream` for decompression.
    ///
    /// This is equivalent to calling:
    /// * `reset(SessionOnly)`
    /// * `disable_dictionary()`
    ///
    /// Wraps the `ZSTD_initCStream()` function.
    pub fn init(&mut self) -> SafeResult {
        let code = unsafe { zstd_sys::ZSTD_initDStream(self.0.as_ptr()) };
        parse_code(code)
    }

    /// Wraps the `ZSTD_initDStream_usingDict()` function.
    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    #[deprecated]
    pub fn init_using_dict(&mut self, dict: &[u8]) -> SafeResult {
        let code = unsafe {
            zstd_sys::ZSTD_initDStream_usingDict(
                self.0.as_ptr(),
                ptr_void(dict),
                dict.len(),
            )
        };
        parse_code(code)
    }

    /// Wraps the `ZSTD_initDStream_usingDDict()` function.
    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    #[deprecated]
    pub fn init_using_ddict<'b>(&mut self, ddict: &DDict<'b>) -> SafeResult
    where
        'b: 'a,
    {
        let code = unsafe {
            zstd_sys::ZSTD_initDStream_usingDDict(
                self.0.as_ptr(),
                ddict.0.as_ptr(),
            )
        };
        parse_code(code)
    }

    /// Resets the state of the context.
    ///
    /// Depending on the reset mode, it can reset the session, the parameters, or both.
    ///
    /// Wraps the `ZSTD_DCtx_reset()` function.
    pub fn reset(&mut self, reset: ResetDirective) -> SafeResult {
        parse_code(unsafe {
            zstd_sys::ZSTD_DCtx_reset(self.0.as_ptr(), reset.as_sys())
        })
    }

    /// Loads a dictionary.
    ///
    /// This will let this context decompress frames that were compressed using this dictionary.
    ///
    /// The dictionary content will be copied internally and does not need to be kept alive after
    /// calling this function.
    ///
    /// If you need to use the same dictionary for multiple contexts, it may be more efficient to
    /// create a `DDict` first, then loads that.
    ///
    /// The dictionary will apply to all future frames, until a new dictionary is set.
    pub fn load_dictionary(&mut self, dict: &[u8]) -> SafeResult {
        parse_code(unsafe {
            zstd_sys::ZSTD_DCtx_loadDictionary(
                self.0.as_ptr(),
                ptr_void(dict),
                dict.len(),
            )
        })
    }

    /// Return to "no-dictionary" mode.
    ///
    /// This will disable any dictionary/prefix previously registered for future frames.
    pub fn disable_dictionary(&mut self) -> SafeResult {
        parse_code(unsafe {
            zstd_sys::ZSTD_DCtx_loadDictionary(
                self.0.as_ptr(),
                core::ptr::null(),
                0,
            )
        })
    }

    /// References a dictionary.
    ///
    /// This will let this context decompress frames compressed with the same dictionary.
    ///
    /// It will apply to all frames decompressed by this context (until a new dictionary is set).
    ///
    /// Wraps the `ZSTD_DCtx_refDDict()` function.
    pub fn ref_ddict<'b>(&mut self, ddict: &DDict<'b>) -> SafeResult
    where
        'b: 'a,
    {
        parse_code(unsafe {
            zstd_sys::ZSTD_DCtx_refDDict(self.0.as_ptr(), ddict.0.as_ptr())
        })
    }

    /// Use some prefix as single-use dictionary for the next frame.
    ///
    /// Just like a dictionary, this only works if compression was done with the same prefix.
    ///
    /// But unlike a dictionary, this only applies to the next frame.
    ///
    /// Wraps the `ZSTD_DCtx_refPrefix()` function.
    pub fn ref_prefix<'b>(&mut self, prefix: &'b [u8]) -> SafeResult
    where
        'b: 'a,
    {
        parse_code(unsafe {
            zstd_sys::ZSTD_DCtx_refPrefix(
                self.0.as_ptr(),
                ptr_void(prefix),
                prefix.len(),
            )
        })
    }

    /// Sets a decompression parameter.
    pub fn set_parameter(&mut self, param: DParameter) -> SafeResult {
        #[cfg(feature = "experimental")]
        use zstd_sys::ZSTD_dParameter::{
            ZSTD_d_experimentalParam1 as ZSTD_d_format,
            ZSTD_d_experimentalParam2 as ZSTD_d_stableOutBuffer,
            ZSTD_d_experimentalParam3 as ZSTD_d_forceIgnoreChecksum,
            ZSTD_d_experimentalParam4 as ZSTD_d_refMultipleDDicts,
        };

        use zstd_sys::ZSTD_dParameter::*;
        use DParameter::*;

        let (param, value) = match param {
            #[cfg(feature = "experimental")]
            Format(format) => (ZSTD_d_format, format as c_int),
            #[cfg(feature = "experimental")]
            StableOutBuffer(stable) => {
                (ZSTD_d_stableOutBuffer, stable as c_int)
            }
            #[cfg(feature = "experimental")]
            ForceIgnoreChecksum(force) => {
                (ZSTD_d_forceIgnoreChecksum, force as c_int)
            }
            #[cfg(feature = "experimental")]
            RefMultipleDDicts(value) => {
                (ZSTD_d_refMultipleDDicts, value as c_int)
            }

            WindowLogMax(value) => (ZSTD_d_windowLogMax, value as c_int),
        };

        parse_code(unsafe {
            zstd_sys::ZSTD_DCtx_setParameter(self.0.as_ptr(), param, value)
        })
    }

    /// Performs a step of a streaming decompression operation.
    ///
    /// This will read some data from `input` and/or write some data to `output`.
    ///
    /// # Returns
    ///
    /// * `Ok(0)` if the current frame just finished decompressing successfully.
    /// * `Ok(hint)` with a hint for the "ideal" amount of input data to provide in the next call.
    ///     Can be safely ignored.
    ///
    /// Wraps the `ZSTD_decompressStream()` function.
    pub fn decompress_stream<C: WriteBuf + ?Sized>(
        &mut self,
        output: &mut OutBuffer<'_, C>,
        input: &mut InBuffer<'_>,
    ) -> SafeResult {
        let mut output = output.wrap();
        let mut input = input.wrap();
        let code = unsafe {
            zstd_sys::ZSTD_decompressStream(
                self.0.as_ptr(),
                ptr_mut(&mut output),
                ptr_mut(&mut input),
            )
        };
        parse_code(code)
    }

    /// Wraps the `ZSTD_DStreamInSize()` function.
    ///
    /// Returns a hint for the recommended size of the input buffer for decompression.
    pub fn in_size() -> usize {
        unsafe { zstd_sys::ZSTD_DStreamInSize() }
    }

    /// Wraps the `ZSTD_DStreamOutSize()` function.
    ///
    /// Returns a hint for the recommended size of the output buffer for decompression.
    pub fn out_size() -> usize {
        unsafe { zstd_sys::ZSTD_DStreamOutSize() }
    }

    /// Wraps the `ZSTD_sizeof_DCtx()` function.
    pub fn sizeof(&self) -> usize {
        unsafe { zstd_sys::ZSTD_sizeof_DCtx(self.0.as_ptr()) }
    }

    /// Wraps the `ZSTD_decompressBlock()` function.
    ///
    /// # Safety
    ///
    /// The bytes written to `dst` become this context's history window, so `dst` must stay
    /// allocated and unmodified until the next call to `decompress_block` or `insert_block` on this
    /// context (or until this context is dropped), as the following block is decoded against it.
    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    pub unsafe fn decompress_block<C: WriteBuf + ?Sized>(
        &mut self,
        dst: &mut C,
        src: &[u8],
    ) -> SafeResult {
        unsafe {
            dst.write_from(|buffer, capacity| {
                parse_code(zstd_sys::ZSTD_decompressBlock(
                    self.0.as_ptr(),
                    buffer,
                    capacity,
                    ptr_void(src),
                    src.len(),
                ))
            })
        }
    }

    /// Wraps the `ZSTD_insertBlock()` function.
    ///
    /// # Safety
    ///
    /// `block` becomes this context's history window, so it must stay allocated and unmodified
    /// until the next call to `decompress_block` or `insert_block` on this context (or until this
    /// context is dropped), as the following block is decoded against it.
    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    pub unsafe fn insert_block(&mut self, block: &[u8]) -> usize {
        unsafe {
            zstd_sys::ZSTD_insertBlock(
                self.0.as_ptr(),
                ptr_void(block),
                block.len(),
            )
        }
    }

    /// Creates a copy of this context.
    ///
    /// This only works before any data has been decompressed. An error will be
    /// returned otherwise.
    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    pub fn try_clone(&self) -> Result<Self, ErrorCode> {
        let context = NonNull::new(unsafe { zstd_sys::ZSTD_createDCtx() })
            .ok_or(0usize)?;

        unsafe { zstd_sys::ZSTD_copyDCtx(context.as_ptr(), self.0.as_ptr()) };

        Ok(DCtx(context, self.1))
    }
}

impl Drop for DCtx<'_> {
    fn drop(&mut self) {
        unsafe {
            zstd_sys::ZSTD_freeDCtx(self.0.as_ptr());
        }
    }
}

unsafe impl Send for DCtx<'_> {}
// Non thread-safe methods already take `&mut self`, so it's fine to implement Sync here.
unsafe impl Sync for DCtx<'_> {}

/// Compression dictionary.
pub struct CDict<'a>(NonNull<zstd_sys::ZSTD_CDict>, PhantomData<&'a ()>);

impl CDict<'static> {
    /// Prepare a dictionary to compress data.
    ///
    /// This will make it easier for compression contexts to load this dictionary.
    ///
    /// The dictionary content will be copied internally, and does not need to be kept around.
    ///
    /// # Panics
    ///
    /// If loading this dictionary failed.
    pub fn create(
        dict_buffer: &[u8],
        compression_level: CompressionLevel,
    ) -> Self {
        Self::try_create(dict_buffer, compression_level)
            .expect("zstd returned null pointer when creating dict")
    }

    /// Prepare a dictionary to compress data.
    ///
    /// This will make it easier for compression contexts to load this dictionary.
    ///
    /// The dictionary content will be copied internally, and does not need to be kept around.
    pub fn try_create(
        dict_buffer: &[u8],
        compression_level: CompressionLevel,
    ) -> Option<Self> {
        Some(CDict(
            NonNull::new(unsafe {
                zstd_sys::ZSTD_createCDict(
                    ptr_void(dict_buffer),
                    dict_buffer.len(),
                    compression_level,
                )
            })?,
            PhantomData,
        ))
    }
}

impl<'a> CDict<'a> {
    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    pub fn create_by_reference(
        dict_buffer: &'a [u8],
        compression_level: CompressionLevel,
    ) -> Self {
        CDict(
            NonNull::new(unsafe {
                zstd_sys::ZSTD_createCDict_byReference(
                    ptr_void(dict_buffer),
                    dict_buffer.len(),
                    compression_level,
                )
            })
            .expect("zstd returned null pointer"),
            PhantomData,
        )
    }

    /// Returns the _current_ memory usage of this dictionary.
    ///
    /// Note that this may change over time.
    pub fn sizeof(&self) -> usize {
        unsafe { zstd_sys::ZSTD_sizeof_CDict(self.0.as_ptr()) }
    }

    /// Returns the dictionary ID for this dict.
    ///
    /// Returns `None` if this dictionary is empty or invalid.
    pub fn get_dict_id(&self) -> Option<NonZeroU32> {
        NonZeroU32::new(unsafe {
            zstd_sys::ZSTD_getDictID_fromCDict(self.0.as_ptr()) as u32
        })
    }
}

/// Wraps the `ZSTD_createCDict()` function.
pub fn create_cdict(
    dict_buffer: &[u8],
    compression_level: CompressionLevel,
) -> CDict<'static> {
    CDict::create(dict_buffer, compression_level)
}

impl<'a> Drop for CDict<'a> {
    fn drop(&mut self) {
        unsafe {
            zstd_sys::ZSTD_freeCDict(self.0.as_ptr());
        }
    }
}

unsafe impl<'a> Send for CDict<'a> {}
unsafe impl<'a> Sync for CDict<'a> {}

/// Wraps the `ZSTD_compress_usingCDict()` function.
pub fn compress_using_cdict(
    cctx: &mut CCtx<'_>,
    dst: &mut [u8],
    src: &[u8],
    cdict: &CDict<'_>,
) -> SafeResult {
    cctx.compress_using_cdict(dst, src, cdict)
}

/// A digested decompression dictionary.
pub struct DDict<'a>(NonNull<zstd_sys::ZSTD_DDict>, PhantomData<&'a ()>);

impl DDict<'static> {
    pub fn create(dict_buffer: &[u8]) -> Self {
        Self::try_create(dict_buffer)
            .expect("zstd returned null pointer when creating dict")
    }

    pub fn try_create(dict_buffer: &[u8]) -> Option<Self> {
        Some(DDict(
            NonNull::new(unsafe {
                zstd_sys::ZSTD_createDDict(
                    ptr_void(dict_buffer),
                    dict_buffer.len(),
                )
            })?,
            PhantomData,
        ))
    }
}

impl<'a> DDict<'a> {
    pub fn sizeof(&self) -> usize {
        unsafe { zstd_sys::ZSTD_sizeof_DDict(self.0.as_ptr()) }
    }

    /// Wraps the `ZSTD_createDDict_byReference()` function.
    ///
    /// The dictionary will keep referencing `dict_buffer`.
    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    pub fn create_by_reference(dict_buffer: &'a [u8]) -> Self {
        DDict(
            NonNull::new(unsafe {
                zstd_sys::ZSTD_createDDict_byReference(
                    ptr_void(dict_buffer),
                    dict_buffer.len(),
                )
            })
            .expect("zstd returned null pointer"),
            PhantomData,
        )
    }

    /// Returns the dictionary ID for this dict.
    ///
    /// Returns `None` if this dictionary is empty or invalid.
    pub fn get_dict_id(&self) -> Option<NonZeroU32> {
        NonZeroU32::new(unsafe {
            zstd_sys::ZSTD_getDictID_fromDDict(self.0.as_ptr()) as u32
        })
    }
}

/// Wraps the `ZSTD_createDDict()` function.
///
/// It copies the dictionary internally, so the resulting `DDict` is `'static`.
pub fn create_ddict(dict_buffer: &[u8]) -> DDict<'static> {
    DDict::create(dict_buffer)
}

impl<'a> Drop for DDict<'a> {
    fn drop(&mut self) {
        unsafe {
            zstd_sys::ZSTD_freeDDict(self.0.as_ptr());
        }
    }
}

unsafe impl<'a> Send for DDict<'a> {}
unsafe impl<'a> Sync for DDict<'a> {}

/// A shared thread pool for one or more compression contexts
#[cfg(all(feature = "experimental", feature = "zstdmt"))]
#[cfg_attr(
    feature = "doc-cfg",
    doc(cfg(all(feature = "experimental", feature = "zstdmt")))
)]
pub struct ThreadPool(NonNull<zstd_sys::ZSTD_threadPool>);

#[cfg(all(feature = "experimental", feature = "zstdmt"))]
#[cfg_attr(
    feature = "doc-cfg",
    doc(cfg(all(feature = "experimental", feature = "zstdmt")))
)]
impl ThreadPool {
    /// Create a thread pool with the specified number of threads.
    ///
    /// # Panics
    ///
    /// If creating the thread pool failed.
    pub fn new(num_threads: usize) -> Self {
        Self::try_new(num_threads)
            .expect("zstd returned null pointer when creating thread pool")
    }

    /// Create a thread pool with the specified number of threads.
    pub fn try_new(num_threads: usize) -> Option<Self> {
        Some(Self(NonNull::new(unsafe {
            zstd_sys::ZSTD_createThreadPool(num_threads)
        })?))
    }
}

#[cfg(all(feature = "experimental", feature = "zstdmt"))]
#[cfg_attr(
    feature = "doc-cfg",
    doc(cfg(all(feature = "experimental", feature = "zstdmt")))
)]
impl Drop for ThreadPool {
    fn drop(&mut self) {
        unsafe {
            zstd_sys::ZSTD_freeThreadPool(self.0.as_ptr());
        }
    }
}

#[cfg(all(feature = "experimental", feature = "zstdmt"))]
#[cfg_attr(
    feature = "doc-cfg",
    doc(cfg(all(feature = "experimental", feature = "zstdmt")))
)]
unsafe impl Send for ThreadPool {}
#[cfg(all(feature = "experimental", feature = "zstdmt"))]
#[cfg_attr(
    feature = "doc-cfg",
    doc(cfg(all(feature = "experimental", feature = "zstdmt")))
)]
unsafe impl Sync for ThreadPool {}

/// Wraps the `ZSTD_decompress_usingDDict()` function.
pub fn decompress_using_ddict(
    dctx: &mut DCtx<'_>,
    dst: &mut [u8],
    src: &[u8],
    ddict: &DDict<'_>,
) -> SafeResult {
    dctx.decompress_using_ddict(dst, src, ddict)
}

/// Compression stream.
///
/// Same as `CCtx`.
pub type CStream<'a> = CCtx<'a>;

// CStream can't be shared across threads, so it does not implement Sync.

/// Allocates a new `CStream`.
pub fn create_cstream<'a>() -> CStream<'a> {
    CCtx::create()
}

/// Prepares an existing `CStream` for compression at the given level.
pub fn init_cstream(
    zcs: &mut CStream<'_>,
    compression_level: CompressionLevel,
) -> SafeResult {
    zcs.init(compression_level)
}

#[derive(Debug)]
/// Wrapper around an input buffer.
///
/// Bytes will be read starting at `src[pos]`.
///
/// `pos` will be updated after reading.
pub struct InBuffer<'a> {
    pub src: &'a [u8],
    pub pos: usize,
}

/// Describe a bytes container, like `Vec<u8>`.
///
/// Represents a contiguous segment of allocated memory, a prefix of which is initialized.
///
/// It allows starting from an uninitializes chunk of memory and writing to it, progressively
/// initializing it. No re-allocation typically occur after the initial creation.
///
/// The main implementors are:
/// * `Vec<u8>` and similar structures. These hold both a length (initialized data) and a capacity
///   (allocated memory).
///
///   Use `Vec::with_capacity` to create an empty `Vec` with non-zero capacity, and the length
///   field will be updated to cover the data written to it (as long as it fits in the given
///   capacity).
/// * `[u8]` and `[u8; N]`. These must start already-initialized, and will not be resized. It will
///   be up to the caller to only use the part that was written (as returned by the various writing
///   operations).
/// * `std::io::Cursor<T: WriteBuf>`. This will ignore data before the cursor's position, and
///   append data after that.
pub unsafe trait WriteBuf {
    /// Returns the valid data part of this container. Should only cover initialized data.
    fn as_slice(&self) -> &[u8];

    /// Returns the full capacity of this container. May include uninitialized data.
    fn capacity(&self) -> usize;

    /// Returns a pointer to the start of the data.
    fn as_mut_ptr(&mut self) -> *mut u8;

    /// Indicates that the first `n` bytes of the container have been written.
    ///
    /// Safety: this should only be called if the `n` first bytes of this buffer have actually been
    /// initialized.
    unsafe fn filled_until(&mut self, n: usize);

    /// Call the given closure using the pointer and capacity from `self`.
    ///
    /// Assumes the given function returns a parseable code, which if valid, represents how many
    /// bytes were written to `self`.
    ///
    /// The given closure must treat its first argument as pointing to potentially uninitialized
    /// memory, and should not read from it.
    ///
    /// In addition, it must have written at least `n` bytes contiguously from this pointer, where
    /// `n` is the returned value.
    unsafe fn write_from<F>(&mut self, f: F) -> SafeResult
    where
        F: FnOnce(*mut c_void, usize) -> SafeResult,
    {
        let res = f(ptr_mut_void(self), self.capacity());
        if let Ok(n) = res {
            self.filled_until(n);
        }
        res
    }
}

#[cfg(feature = "std")]
#[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "std")))]
unsafe impl<T> WriteBuf for std::io::Cursor<T>
where
    T: WriteBuf,
{
    fn as_slice(&self) -> &[u8] {
        &self.get_ref().as_slice()[self.position() as usize..]
    }

    fn capacity(&self) -> usize {
        self.get_ref()
            .capacity()
            .saturating_sub(self.position() as usize)
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        let start = self.position() as usize;
        assert!(start <= self.get_ref().capacity());
        // Safety: start is still in the same memory allocation
        unsafe { self.get_mut().as_mut_ptr().add(start) }
    }

    unsafe fn filled_until(&mut self, n: usize) {
        // Early exit: `n = 0` does not indicate anything.
        if n == 0 {
            return;
        }

        // Here we assume data _before_ self.position() was already initialized.
        // Egh it's not actually guaranteed by Cursor? So let's guarantee it ourselves.
        // Since the cursor wraps another `WriteBuf`, we know how much data is initialized there.
        let position = self.position() as usize;
        let initialized = self.get_ref().as_slice().len();
        if let Some(uninitialized) = position.checked_sub(initialized) {
            // Here, the cursor is further than the known-initialized part.
            // Cursor's solution is to pad with zeroes, so let's do the same.
            // We'll zero bytes from the end of valid data (as_slice().len()) to the cursor position.

            // Safety:
            // * We know `n > 0`
            // * This means `self.capacity() > 0` (promise by the caller)
            // * This means `self.get_ref().capacity() > self.position`
            // * This means that `position` is within the nested pointer's allocation.
            // * Finally, `initialized + uninitialized = position`, so the entire byte
            //   range here is within the allocation
            unsafe {
                self.get_mut()
                    .as_mut_ptr()
                    .add(initialized)
                    .write_bytes(0u8, uninitialized)
            };
        }

        let start = self.position() as usize;
        assert!(start + n <= self.get_ref().capacity());
        self.get_mut().filled_until(start + n);
    }
}

#[cfg(feature = "std")]
#[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "std")))]
unsafe impl<'a> WriteBuf for &'a mut std::vec::Vec<u8> {
    fn as_slice(&self) -> &[u8] {
        std::vec::Vec::as_slice(self)
    }

    fn capacity(&self) -> usize {
        std::vec::Vec::capacity(self)
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        std::vec::Vec::as_mut_ptr(self)
    }

    unsafe fn filled_until(&mut self, n: usize) {
        std::vec::Vec::set_len(self, n)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "std")))]
unsafe impl WriteBuf for std::vec::Vec<u8> {
    fn as_slice(&self) -> &[u8] {
        &self[..]
    }
    fn capacity(&self) -> usize {
        self.capacity()
    }
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.as_mut_ptr()
    }
    unsafe fn filled_until(&mut self, n: usize) {
        self.set_len(n);
    }
}

#[cfg(feature = "arrays")]
#[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "arrays")))]
unsafe impl<const N: usize> WriteBuf for [u8; N] {
    fn as_slice(&self) -> &[u8] {
        self
    }
    fn capacity(&self) -> usize {
        self.len()
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        (&mut self[..]).as_mut_ptr()
    }

    unsafe fn filled_until(&mut self, _n: usize) {
        // Assume the slice is already initialized
    }
}

unsafe impl WriteBuf for [u8] {
    fn as_slice(&self) -> &[u8] {
        self
    }
    fn capacity(&self) -> usize {
        self.len()
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.as_mut_ptr()
    }

    unsafe fn filled_until(&mut self, _n: usize) {
        // Assume the slice is already initialized
    }
}

/*
// This is possible, but... why?
unsafe impl<'a> WriteBuf for OutBuffer<'a, [u8]> {
    fn as_slice(&self) -> &[u8] {
        self.dst
    }
    fn capacity(&self) -> usize {
        self.dst.len()
    }
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.dst.as_mut_ptr()
    }
    unsafe fn filled_until(&mut self, n: usize) {
        self.pos = n;
    }
}
*/

#[derive(Debug)]
/// Wrapper around an output buffer.
///
/// `C` is usually either `[u8]` or `Vec<u8>`.
///
/// Bytes will be written starting at `dst[pos]`.
///
/// `pos` will be updated after writing.
///
/// # Invariant
///
/// `pos <= dst.capacity()`
pub struct OutBuffer<'a, C: WriteBuf + ?Sized> {
    dst: &'a mut C,
    pos: usize,
}

/// Convenience method to get a mut pointer from a mut ref.
fn ptr_mut<B>(ptr_void: &mut B) -> *mut B {
    ptr_void as *mut B
}

/// Interface between a C-level ZSTD_outBuffer and a rust-level `OutBuffer`.
///
/// Will update the parent buffer from the C buffer on drop.
struct OutBufferWrapper<'a, 'b, C: WriteBuf + ?Sized> {
    buf: zstd_sys::ZSTD_outBuffer,
    parent: &'a mut OutBuffer<'b, C>,
}

impl<'a, 'b: 'a, C: WriteBuf + ?Sized> Deref for OutBufferWrapper<'a, 'b, C> {
    type Target = zstd_sys::ZSTD_outBuffer;

    fn deref(&self) -> &Self::Target {
        &self.buf
    }
}

impl<'a, 'b: 'a, C: WriteBuf + ?Sized> DerefMut
    for OutBufferWrapper<'a, 'b, C>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buf
    }
}

impl<'a, C: WriteBuf + ?Sized> OutBuffer<'a, C> {
    /// Returns a new `OutBuffer` around the given slice.
    ///
    /// Starts with `pos = 0`.
    pub fn around(dst: &'a mut C) -> Self {
        OutBuffer { dst, pos: 0 }
    }

    /// Returns a new `OutBuffer` around the given slice, starting at the given position.
    ///
    /// # Panics
    ///
    /// If `pos > dst.capacity()`.
    pub fn around_pos(dst: &'a mut C, pos: usize) -> Self {
        if pos > dst.capacity() {
            panic!("Given position outside of the buffer bounds.");
        }

        OutBuffer { dst, pos }
    }

    /// Returns the current cursor position.
    ///
    /// Guaranteed to be <= self.capacity()
    pub fn pos(&self) -> usize {
        assert!(self.pos <= self.dst.capacity());
        self.pos
    }

    /// Returns the capacity of the underlying buffer.
    pub fn capacity(&self) -> usize {
        self.dst.capacity()
    }

    /// Sets the new cursor position.
    ///
    /// # Panics
    ///
    /// If `pos > self.dst.capacity()`.
    ///
    /// # Safety
    ///
    /// Data up to `pos` must have actually been written to.
    pub unsafe fn set_pos(&mut self, pos: usize) {
        if pos > self.dst.capacity() {
            panic!("Given position outside of the buffer bounds.");
        }

        self.dst.filled_until(pos);

        self.pos = pos;
    }

    fn wrap<'b>(&'b mut self) -> OutBufferWrapper<'b, 'a, C> {
        OutBufferWrapper {
            buf: zstd_sys::ZSTD_outBuffer {
                dst: ptr_mut_void(self.dst),
                size: self.dst.capacity(),
                pos: self.pos,
            },
            parent: self,
        }
    }

    /// Returns the part of this buffer that was written to.
    pub fn as_slice<'b>(&'b self) -> &'a [u8]
    where
        'b: 'a,
    {
        let pos = self.pos;
        &self.dst.as_slice()[..pos]
    }

    /// Returns a pointer to the start of this buffer.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.dst.as_mut_ptr()
    }
}

impl<'a, 'b, C: WriteBuf + ?Sized> Drop for OutBufferWrapper<'a, 'b, C> {
    fn drop(&mut self) {
        // Safe because we guarantee that data until `self.buf.pos` has been written.
        unsafe { self.parent.set_pos(self.buf.pos) };
    }
}

struct InBufferWrapper<'a, 'b> {
    buf: zstd_sys::ZSTD_inBuffer,
    parent: &'a mut InBuffer<'b>,
}

impl<'a, 'b: 'a> Deref for InBufferWrapper<'a, 'b> {
    type Target = zstd_sys::ZSTD_inBuffer;

    fn deref(&self) -> &Self::Target {
        &self.buf
    }
}

impl<'a, 'b: 'a> DerefMut for InBufferWrapper<'a, 'b> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buf
    }
}

impl<'a> InBuffer<'a> {
    /// Returns a new `InBuffer` around the given slice.
    ///
    /// Starts with `pos = 0`.
    pub fn around(src: &'a [u8]) -> Self {
        InBuffer { src, pos: 0 }
    }

    /// Returns the current cursor position.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Sets the new cursor position.
    ///
    /// # Panics
    ///
    /// If `pos > self.src.len()`.
    pub fn set_pos(&mut self, pos: usize) {
        if pos > self.src.len() {
            panic!("Given position outside of the buffer bounds.");
        }
        self.pos = pos;
    }

    fn wrap<'b>(&'b mut self) -> InBufferWrapper<'b, 'a> {
        InBufferWrapper {
            buf: zstd_sys::ZSTD_inBuffer {
                src: ptr_void(self.src),
                size: self.src.len(),
                pos: self.pos,
            },
            parent: self,
        }
    }
}

impl<'a, 'b> Drop for InBufferWrapper<'a, 'b> {
    fn drop(&mut self) {
        self.parent.set_pos(self.buf.pos);
    }
}

/// A Decompression stream.
///
/// Same as `DCtx`.
pub type DStream<'a> = DCtx<'a>;

// Some functions work on a "frame prefix".
// TODO: Define `struct FramePrefix(&[u8]);` and move these functions to it?
//
// Some other functions work on a dictionary (not CDict or DDict).
// Same thing?

/// Wraps the `ZSTD_findFrameCompressedSize()` function.
///
/// `src` should contain at least an entire frame.
pub fn find_frame_compressed_size(src: &[u8]) -> SafeResult {
    let code = unsafe {
        zstd_sys::ZSTD_findFrameCompressedSize(ptr_void(src), src.len())
    };
    parse_code(code)
}

/// Wraps the `ZSTD_getFrameContentSize()` function.
///
/// Args:
/// * `src`: A prefix of the compressed frame. It should at least include the frame header.
///
/// Returns:
/// * `Err(ContentSizeError)` if `src` is too small of a prefix, or if it appears corrupted.
/// * `Ok(None)` if the frame does not include a content size.
/// * `Ok(Some(content_size_in_bytes))` otherwise.
pub fn get_frame_content_size(
    src: &[u8],
) -> Result<Option<u64>, ContentSizeError> {
    parse_content_size(unsafe {
        zstd_sys::ZSTD_getFrameContentSize(ptr_void(src), src.len())
    })
}

/// Wraps the `ZSTD_findDecompressedSize()` function.
///
/// `src` should be exactly a sequence of ZSTD frames.
#[cfg(feature = "experimental")]
#[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
pub fn find_decompressed_size(
    src: &[u8],
) -> Result<Option<u64>, ContentSizeError> {
    parse_content_size(unsafe {
        zstd_sys::ZSTD_findDecompressedSize(ptr_void(src), src.len())
    })
}

/// Wraps the `ZSTD_isFrame()` function.
#[cfg(feature = "experimental")]
#[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
pub fn is_frame(buffer: &[u8]) -> bool {
    unsafe { zstd_sys::ZSTD_isFrame(ptr_void(buffer), buffer.len()) > 0 }
}

/// Wraps the `ZSTD_getDictID_fromDict()` function.
///
/// Returns `None` if the dictionary is not a valid zstd dictionary.
pub fn get_dict_id_from_dict(dict: &[u8]) -> Option<NonZeroU32> {
    NonZeroU32::new(unsafe {
        zstd_sys::ZSTD_getDictID_fromDict(ptr_void(dict), dict.len()) as u32
    })
}

/// Wraps the `ZSTD_getDictID_fromFrame()` function.
///
/// Returns `None` if the dictionary ID could not be decoded. This may happen if:
/// * The frame was not encoded with a dictionary.
/// * The frame intentionally did not include dictionary ID.
/// * The dictionary was non-conformant.
/// * `src` is too small and does not include the frame header.
/// * `src` is not a valid zstd frame prefix.
pub fn get_dict_id_from_frame(src: &[u8]) -> Option<NonZeroU32> {
    NonZeroU32::new(unsafe {
        zstd_sys::ZSTD_getDictID_fromFrame(ptr_void(src), src.len()) as u32
    })
}

/// What kind of context reset should be applied.
pub enum ResetDirective {
    /// Only the session will be reset.
    ///
    /// All parameters will be preserved (including the dictionary).
    /// But any frame being processed will be dropped.
    ///
    /// It can be useful to start re-using a context after an error or when an
    /// ongoing compression is no longer needed.
    SessionOnly,

    /// Only reset parameters (including dictionary or referenced prefix).
    ///
    /// All parameters will be reset to default values.
    ///
    /// This can only be done between sessions - no compression or decompression must be ongoing.
    Parameters,

    /// Reset both the session and parameters.
    ///
    /// The result is similar to a newly created context.
    SessionAndParameters,
}

impl ResetDirective {
    fn as_sys(self) -> zstd_sys::ZSTD_ResetDirective {
        match self {
            ResetDirective::SessionOnly => zstd_sys::ZSTD_ResetDirective::ZSTD_reset_session_only,
            ResetDirective::Parameters => zstd_sys::ZSTD_ResetDirective::ZSTD_reset_parameters,
            ResetDirective::SessionAndParameters => zstd_sys::ZSTD_ResetDirective::ZSTD_reset_session_and_parameters,
        }
    }
}

#[cfg(feature = "experimental")]
#[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum FrameFormat {
    /// Regular zstd format.
    One = zstd_sys::ZSTD_format_e::ZSTD_f_zstd1 as u32,

    /// Skip the 4 bytes identifying the content as zstd-compressed data.
    Magicless = zstd_sys::ZSTD_format_e::ZSTD_f_zstd1_magicless as u32,
}

#[cfg(feature = "experimental")]
#[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum DictAttachPref {
    DefaultAttach =
        zstd_sys::ZSTD_dictAttachPref_e::ZSTD_dictDefaultAttach as u32,
    ForceAttach = zstd_sys::ZSTD_dictAttachPref_e::ZSTD_dictForceAttach as u32,
    ForceCopy = zstd_sys::ZSTD_dictAttachPref_e::ZSTD_dictForceCopy as u32,
    ForceLoad = zstd_sys::ZSTD_dictAttachPref_e::ZSTD_dictForceLoad as u32,
}

#[cfg(feature = "experimental")]
#[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ParamSwitch {
    Auto = zstd_sys::ZSTD_ParamSwitch_e::ZSTD_ps_auto as u32,
    Enable = zstd_sys::ZSTD_ParamSwitch_e::ZSTD_ps_enable as u32,
    Disable = zstd_sys::ZSTD_ParamSwitch_e::ZSTD_ps_disable as u32,
}

/// A compression parameter.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CParameter {
    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    RSyncable(bool),

    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    Format(FrameFormat),

    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    ForceMaxWindow(bool),

    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    ForceAttachDict(DictAttachPref),

    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    LiteralCompressionMode(ParamSwitch),

    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    SrcSizeHint(u32),

    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    EnableDedicatedDictSearch(bool),

    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    StableInBuffer(bool),

    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    StableOutBuffer(bool),

    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    BlockDelimiters(bool),

    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    ValidateSequences(bool),

    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    UseBlockSplitter(ParamSwitch),

    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    UseRowMatchFinder(ParamSwitch),

    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    DeterministicRefPrefix(bool),

    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    PrefetchCDictTables(ParamSwitch),

    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    EnableSeqProducerFallback(bool),

    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    MaxBlockSize(u32),

    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    SearchForExternalRepcodes(ParamSwitch),

    /// Target CBlock size.
    ///
    /// Tries to make compressed blocks fit in this size (not a guarantee, just a target).
    /// Useful to reduce end-to-end latency in low-bandwidth environments.
    ///
    /// No target when the value is 0.
    TargetCBlockSize(u32),

    /// Compression level to use.
    ///
    /// Compression levels are global presets for the other compression parameters.
    CompressionLevel(CompressionLevel),

    /// Maximum allowed back-reference distance.
    ///
    /// The actual distance is 2 power "this value".
    WindowLog(u32),

    HashLog(u32),

    ChainLog(u32),

    SearchLog(u32),

    MinMatch(u32),

    TargetLength(u32),

    Strategy(Strategy),

    EnableLongDistanceMatching(bool),

    LdmHashLog(u32),

    LdmMinMatch(u32),

    LdmBucketSizeLog(u32),

    LdmHashRateLog(u32),

    ContentSizeFlag(bool),

    ChecksumFlag(bool),

    DictIdFlag(bool),

    /// How many threads will be spawned.
    ///
    /// With a default value of `0`, `compress_stream*` functions block until they complete.
    ///
    /// With any other value (including 1, a single compressing thread), these methods directly
    /// return, and the actual compression is done in the background (until a flush is requested).
    ///
    /// Note: this will only work if the `zstdmt` feature is activated.
    NbWorkers(u32),

    /// Size in bytes of a compression job.
    ///
    /// Does not have any effect when `NbWorkers` is set to 0.
    ///
    /// The default value of 0 finds the best job size based on the compression parameters.
    ///
    /// Note: this will only work if the `zstdmt` feature is activated.
    JobSize(u32),

    /// Specifies how much overlap must be given to each worker.
    ///
    /// Possible values:
    ///
    /// * `0` (default value): automatic overlap based on compression strategy.
    /// * `1`: No overlap
    /// * `1 < n < 9`: Overlap a fraction of the window size, defined as `1/(2 ^ 9-n)`.
    /// * `9`: Full overlap (as long as the window)
    /// * `9 < m`: Will return an error.
    ///
    /// Note: this will only work if the `zstdmt` feature is activated.
    OverlapSizeLog(u32),
}

/// A decompression parameter.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DParameter {
    WindowLogMax(u32),

    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    /// See `FrameFormat`.
    Format(FrameFormat),

    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    StableOutBuffer(bool),

    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    ForceIgnoreChecksum(bool),

    #[cfg(feature = "experimental")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
    RefMultipleDDicts(bool),
}

/// Wraps the `ZDICT_trainFromBuffer()` function.
#[cfg(feature = "zdict_builder")]
#[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "zdict_builder")))]
pub fn train_from_buffer<C: WriteBuf + ?Sized>(
    dict_buffer: &mut C,
    samples_buffer: &[u8],
    samples_sizes: &[usize],
) -> SafeResult {
    assert_eq!(samples_buffer.len(), samples_sizes.iter().sum());

    unsafe {
        dict_buffer.write_from(|buffer, capacity| {
            parse_code(zstd_sys::ZDICT_trainFromBuffer(
                buffer,
                capacity,
                ptr_void(samples_buffer),
                samples_sizes.as_ptr(),
                samples_sizes.len() as u32,
            ))
        })
    }
}

/// Wraps the `ZDICT_getDictID()` function.
#[cfg(feature = "zdict_builder")]
#[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "zdict_builder")))]
pub fn get_dict_id(dict_buffer: &[u8]) -> Option<NonZeroU32> {
    NonZeroU32::new(unsafe {
        zstd_sys::ZDICT_getDictID(ptr_void(dict_buffer), dict_buffer.len())
    })
}

/// Wraps the `ZSTD_getBlockSize()` function.
#[cfg(feature = "experimental")]
#[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
pub fn get_block_size(cctx: &CCtx) -> usize {
    unsafe { zstd_sys::ZSTD_getBlockSize(cctx.0.as_ptr()) }
}

/// Wraps the `ZSTD_decompressBound` function
#[cfg(feature = "experimental")]
#[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
pub fn decompress_bound(data: &[u8]) -> Result<u64, ErrorCode> {
    let bound =
        unsafe { zstd_sys::ZSTD_decompressBound(ptr_void(data), data.len()) };
    if is_error(bound as usize) {
        Err(bound as usize)
    } else {
        Ok(bound)
    }
}

/// Given a buffer of size `src_size`, returns the maximum number of sequences that can ge
/// generated.
#[cfg(feature = "experimental")]
#[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
pub fn sequence_bound(src_size: usize) -> usize {
    // Safety: Just FFI.
    unsafe { zstd_sys::ZSTD_sequenceBound(src_size) }
}

/// Returns the minimum extra space when output and input buffer overlap.
///
/// When using in-place decompression, the output buffer must be at least this much bigger (in
/// bytes) than the input buffer. The extra space must be at the front of the output buffer (the
/// input buffer must be at the end of the output buffer).
#[cfg(feature = "experimental")]
#[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "experimental")))]
pub fn decompression_margin(
    compressed_data: &[u8],
) -> Result<usize, ErrorCode> {
    parse_code(unsafe {
        zstd_sys::ZSTD_decompressionMargin(
            ptr_void(compressed_data),
            compressed_data.len(),
        )
    })
}
