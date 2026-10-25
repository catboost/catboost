//! The seekable format splits the compressed data into a series of "frames",
//! each compressed individually so that decompression of a section in the
//! middle of an archive only requires zstd to decompress at most a frame's
//! worth of extra data, instead of the entire archive.

use core::{marker::PhantomData, ptr::NonNull};

use crate::{
    parse_code, ptr_mut, ptr_void, CompressionLevel, InBuffer, OutBuffer,
    SafeResult, WriteBuf, SEEKABLE_FRAMEINDEX_TOOLARGE,
};

/// Indicates that the passed frame index is too large.
///
/// This happens when `frame_index > num_frames()`.
#[derive(Debug, PartialEq)]
pub struct FrameIndexTooLargeError;

impl core::fmt::Display for FrameIndexTooLargeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("Frame index too large")
    }
}

/// Required to tracking streaming operation.
///
/// Streaming objects are reusable to avoid allocation and deallocation,
/// to start a new compression operation call `init()`.
pub struct SeekableCStream(NonNull<zstd_sys::ZSTD_seekable_CStream>);

unsafe impl Send for SeekableCStream {}
unsafe impl Sync for SeekableCStream {}

impl Default for SeekableCStream {
    fn default() -> Self {
        SeekableCStream::create()
    }
}

impl SeekableCStream {
    /// Tries to create a new `SeekableCStream`.
    ///
    /// Returns `None` if zstd returns a NULL pointer - may happen if allocation fails.
    pub fn try_create() -> Option<Self> {
        // Safety: Just FFI
        Some(SeekableCStream(NonNull::new(unsafe {
            zstd_sys::ZSTD_seekable_createCStream()
        })?))
    }

    /// Creates a new `SeekableCStream`.
    ///
    /// # Panics
    ///
    /// If zstd returns a NULL pointer.
    pub fn create() -> Self {
        Self::try_create()
            .expect("zstd returned null pointer when creating new seekable compression stream")
    }

    /// Wraps the `ZSTD_seekable_initCStream()` function.
    ///
    /// Call this to initialize a `SeekableCStream` object for a new compression operation.
    /// - `max_frame_size` indicates the size at which to automatically start a new seekable
    /// frame. `max_frame_size == 0` implies the default maximum size. Smaller frame sizes allow
    /// faster decompression of small segments, since retrieving a single byte requires
    /// decompression of the full frame where the byte belongs. In general, size the frames
    /// to roughly correspond to the access granularity (when it's known). But small sizes
    /// also reduce compression ratio. Avoid really tiny frame sizes (< 1 KB), that would
    /// hurt compression ratio considerably.
    /// - `checksum_flag` indicates whether or not the seek table should include frame
    /// checksums on the uncompressed data for verification.
    ///
    /// Returns a size hint for input to provide for compression, or an error code.
    pub fn init(
        &mut self,
        compression_level: CompressionLevel,
        checksum_flag: bool,
        max_frame_size: u32,
    ) -> SafeResult {
        // Safety: Just FFI
        let code = unsafe {
            zstd_sys::ZSTD_seekable_initCStream(
                self.0.as_ptr(),
                compression_level,
                checksum_flag as i32,
                max_frame_size,
            )
        };
        parse_code(code)
    }

    /// Wraps the `ZSTD_seekable_compressStream()` function.
    ///
    /// Call this repetitively to consume input stream. The function will automatically
    /// update both `pos` fields. Note that it may not consume the entire input, in which
    /// case `pos < size`, and it's up to the caller to present again remaining data.
    ///
    /// Returns a size hint, preferred number of bytes to use as input for the next call
    /// or an error code. Note that it's just a hint, to help latency a little, any other
    /// value will work fine.
    pub fn compress_stream<C: WriteBuf + ?Sized>(
        &mut self,
        output: &mut OutBuffer<'_, C>,
        input: &mut InBuffer<'_>,
    ) -> SafeResult {
        let mut output = output.wrap();
        let mut input = input.wrap();
        // Safety: Just FFI
        let code = unsafe {
            zstd_sys::ZSTD_seekable_compressStream(
                self.0.as_ptr(),
                ptr_mut(&mut output),
                ptr_mut(&mut input),
            )
        };
        parse_code(code)
    }

    /// Wraps the `ZSTD_seekable_endFrame()` function.
    ///
    /// Call this any time to end the current frame and start a new one.
    pub fn end_frame<C: WriteBuf + ?Sized>(
        &mut self,
        output: &mut OutBuffer<'_, C>,
    ) -> SafeResult {
        let mut output = output.wrap();
        // Safety: Just FFI
        let code = unsafe {
            zstd_sys::ZSTD_seekable_endFrame(
                self.0.as_ptr(),
                ptr_mut(&mut output),
            )
        };
        parse_code(code)
    }

    /// Wraps the `ZSTD_seekable_endStream()` function.
    ///
    /// This will end the current frame, and then write the seek table so that
    /// decompressors can efficiently find compressed frames.
    ///
    /// Returns a number > 0 if it was unable to flush all the necessary data to `output`.
    /// In this case, it should be called again until all remaining data is flushed out and
    /// 0 is returned.
    pub fn end_stream<C: WriteBuf + ?Sized>(
        &mut self,
        output: &mut OutBuffer<'_, C>,
    ) -> SafeResult {
        let mut output = output.wrap();
        // Safety: Just FFI
        let code = unsafe {
            zstd_sys::ZSTD_seekable_endStream(
                self.0.as_ptr(),
                ptr_mut(&mut output),
            )
        };
        parse_code(code)
    }
}

impl Drop for SeekableCStream {
    fn drop(&mut self) {
        // Safety: Just FFI
        unsafe {
            zstd_sys::ZSTD_seekable_freeCStream(self.0.as_ptr());
        }
    }
}

/// Allows for the seek table to be constructed directly.
///
/// This table can then be appended to a file of concatenated frames. This allows the
/// frames to be compressed independently, even in parallel, and compiled together
/// afterward into a seekable archive.
pub struct FrameLog(NonNull<zstd_sys::ZSTD_frameLog>);

unsafe impl Send for FrameLog {}
unsafe impl Sync for FrameLog {}

impl FrameLog {
    /// Tries to create a new `FrameLog`.
    ///
    /// Returns `None` if zstd returns a NULL pointer - may happen if allocation fails.
    pub fn try_create(checksum_flag: bool) -> Option<Self> {
        Some(FrameLog(
            // Safety: Just FFI
            NonNull::new(unsafe {
                zstd_sys::ZSTD_seekable_createFrameLog(checksum_flag as i32)
            })?,
        ))
    }

    /// Creates a new `FrameLog`.
    ///
    /// # Panics
    ///
    /// If zstd returns a NULL pointer.
    pub fn create(checksum_flag: bool) -> Self {
        Self::try_create(checksum_flag)
            .expect("Zstd returned null pointer when creating new frame log")
    }

    /// Needs to be called once for each frame in the archive.
    ///
    /// If the `FrameLog` was created with `checksum_flag == false`, the `checksum` may be none
    /// and any value assigned to it will be ignored. If the `FrameLog` was created with
    /// `checksum_flag == true`, it should be the least significant 32 bits of the XXH64
    /// hash of the uncompressed data.
    pub fn log_frame(
        &mut self,
        compressed_size: u32,
        decompressed_size: u32,
        checksum: Option<u32>,
    ) -> SafeResult {
        // Safety: Just FFI
        let code = unsafe {
            zstd_sys::ZSTD_seekable_logFrame(
                self.0.as_ptr(),
                compressed_size,
                decompressed_size,
                checksum.unwrap_or_default(),
            )
        };
        parse_code(code)
    }

    /// Writes the seek table to `output`.
    ///
    /// Returns 0 if the entire table was written. Otherwise, it will be equal to the number
    /// of bytes left to write.
    pub fn write_seek_table<C: WriteBuf + ?Sized>(
        &mut self,
        output: &mut OutBuffer<'_, C>,
    ) -> SafeResult {
        let mut output = output.wrap();
        // Safety: Just FFI
        let code = unsafe {
            zstd_sys::ZSTD_seekable_writeSeekTable(
                self.0.as_ptr(),
                ptr_mut(&mut output),
            )
        };
        parse_code(code)
    }
}

impl Drop for FrameLog {
    fn drop(&mut self) {
        // Safety: Just FFI
        unsafe {
            zstd_sys::ZSTD_seekable_freeFrameLog(self.0.as_ptr());
        }
    }
}

/// A seekable decompression object.
///
/// The lifetime references the potential buffer that holds the data of this seekable.
pub struct Seekable<'a>(NonNull<zstd_sys::ZSTD_seekable>, PhantomData<&'a ()>);

unsafe impl Send for Seekable<'_> {}
unsafe impl Sync for Seekable<'_> {}

impl Default for Seekable<'_> {
    fn default() -> Self {
        Seekable::create()
    }
}

impl<'a> Seekable<'a> {
    /// Tries to create a new `Seekable`.
    ///
    /// Returns `None` if zstd returns a NULL pointer - may happen if allocation fails.
    pub fn try_create() -> Option<Self> {
        // Safety: Just FFI
        Some(Seekable(
            NonNull::new(unsafe { zstd_sys::ZSTD_seekable_create() })?,
            PhantomData,
        ))
    }

    /// Creates a new `Seekable`.
    ///
    /// # Panics
    ///
    /// If zstd returns a NULL pointer.
    pub fn create() -> Self {
        Self::try_create()
            .expect("Zstd returned null pointer when creating new seekable")
    }

    /// Initializes this `Seekable` with the the seek table provided in `src`.
    ///
    /// The data contained in `src` should be the entire seekable file, including the seek table.
    /// Consider using `init_advanced()`, if it not feasible to have the entire seekable file in
    /// memory.
    pub fn init_buff(&mut self, src: &'a [u8]) -> SafeResult {
        // Safety: Just FFI
        let code = unsafe {
            zstd_sys::ZSTD_seekable_initBuff(
                self.0.as_ptr(),
                ptr_void(src),
                src.len(),
            )
        };

        parse_code(code)
    }

    /// Decompresses the length of `dst` at decompressed offset `offset`.
    ///
    /// May have to decompress the entire prefix of the frame before the desired data if it has
    /// not already processed this section. If this is called multiple times for a consecutive
    /// range of data, it will efficiently retain the decompressor object and avoid
    /// redecompressing frame prefixes.
    ///
    /// Returns the number of bytes decompressed, or an error code.
    pub fn decompress<C: WriteBuf + ?Sized>(
        &mut self,
        dst: &mut C,
        offset: u64,
    ) -> SafeResult {
        unsafe {
            dst.write_from(|buffer, capacity| {
                parse_code(zstd_sys::ZSTD_seekable_decompress(
                    self.0.as_ptr(),
                    buffer,
                    capacity,
                    offset,
                ))
            })
        }
    }

    /// Decompresses the frame with index `frame_index` into `dst`.
    ///
    /// Returns an error if `frame_index` is larger than the value returned by `num_frames()`.
    pub fn decompress_frame<C: WriteBuf + ?Sized>(
        &mut self,
        dst: &mut C,
        frame_index: u32,
    ) -> SafeResult {
        unsafe {
            dst.write_from(|buffer, capacity| {
                parse_code(zstd_sys::ZSTD_seekable_decompressFrame(
                    self.0.as_ptr(),
                    buffer,
                    capacity,
                    frame_index,
                ))
            })
        }
    }

    /// Get the number of frames of this seekable object.
    ///
    /// Returns `0` if the seekable is not initialized.
    pub fn num_frames(&self) -> u32 {
        unsafe { zstd_sys::ZSTD_seekable_getNumFrames(self.0.as_ptr()) }
    }

    /// Get the offset of the compressed frame.
    ///
    /// Returns an error if `frame_index` is out of range.
    pub fn frame_compressed_offset(
        &self,
        frame_index: u32,
    ) -> Result<u64, FrameIndexTooLargeError> {
        let offset = unsafe {
            zstd_sys::ZSTD_seekable_getFrameCompressedOffset(
                self.0.as_ptr(),
                frame_index,
            )
        };

        if offset == SEEKABLE_FRAMEINDEX_TOOLARGE {
            return Err(FrameIndexTooLargeError);
        }

        Ok(offset)
    }

    /// Get the offset of the decompressed frame.
    ///
    /// Returns an error if `frame_index` is out of range.
    pub fn frame_decompressed_offset(
        &self,
        frame_index: u32,
    ) -> Result<u64, FrameIndexTooLargeError> {
        let offset = unsafe {
            zstd_sys::ZSTD_seekable_getFrameDecompressedOffset(
                self.0.as_ptr(),
                frame_index,
            )
        };

        if offset == SEEKABLE_FRAMEINDEX_TOOLARGE {
            return Err(FrameIndexTooLargeError);
        }

        Ok(offset)
    }

    /// Get the size of the compressed frame.
    ///
    /// Returns an error if `frame_index` is out of range.
    pub fn frame_compressed_size(&self, frame_index: u32) -> SafeResult {
        let code = unsafe {
            zstd_sys::ZSTD_seekable_getFrameCompressedSize(
                self.0.as_ptr(),
                frame_index,
            )
        };

        parse_code(code)
    }

    /// Get the size of the decompressed frame.
    ///
    /// Returns an error if `frame_index` is out of range.
    pub fn frame_decompressed_size(&self, frame_index: u32) -> SafeResult {
        let code = unsafe {
            zstd_sys::ZSTD_seekable_getFrameDecompressedSize(
                self.0.as_ptr(),
                frame_index,
            )
        };

        parse_code(code)
    }

    /// Get the frame at the given offset.
    pub fn offset_to_frame_index(&self, offset: u64) -> u32 {
        unsafe {
            zstd_sys::ZSTD_seekable_offsetToFrameIndex(self.0.as_ptr(), offset)
        }
    }
}

impl<'a> Drop for Seekable<'a> {
    fn drop(&mut self) {
        // Safety: Just FFI
        unsafe {
            zstd_sys::ZSTD_seekable_free(self.0.as_ptr());
        }
    }
}

#[cfg(feature = "std")]
pub struct AdvancedSeekable<'a, F> {
    inner: Seekable<'a>,
    // We can't use Box<F> since it'd break rust aliasing rules when calling
    // advanced_read/advanced_seek through the C code.
    src: *mut F,
}

#[cfg(feature = "std")]
unsafe impl<F> Send for AdvancedSeekable<'_, F> where F: Send {}
#[cfg(feature = "std")]
unsafe impl<F> Sync for AdvancedSeekable<'_, F> where F: Sync {}

#[cfg(feature = "std")]
impl<'a, F> core::ops::Deref for AdvancedSeekable<'a, F> {
    type Target = Seekable<'a>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[cfg(feature = "std")]
impl<'a, F> core::ops::DerefMut for AdvancedSeekable<'a, F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[cfg(feature = "std")]
impl<'a, F> Drop for AdvancedSeekable<'a, F> {
    fn drop(&mut self) {
        use std::boxed::Box;
        // this drops the box
        let _: Box<F> = unsafe { Box::from_raw(self.src) };
    }
}

impl<'a> Seekable<'a> {
    /// A general API allowing the client to provide its own read and seek implementations.
    ///
    /// Initializes this seekable without having the complete compressed data in memory,
    /// but seeks and reads `src` as required. Use this function if you are looking for
    /// an alternative to the `ZSTD_seekable_initFile()` function.
    #[cfg(feature = "std")]
    #[cfg_attr(feature = "doc-cfg", doc(cfg(feature = "std")))]
    pub fn init_advanced<F>(
        self,
        src: std::boxed::Box<F>,
    ) -> Result<AdvancedSeekable<'a, F>, crate::ErrorCode>
    where
        F: std::io::Read + std::io::Seek,
    {
        let opaque = std::boxed::Box::into_raw(src) as *mut F;
        let custom_file = zstd_sys::ZSTD_seekable_customFile {
            opaque: opaque as *mut core::ffi::c_void,
            read: Some(advanced_read::<F>),
            seek: Some(advanced_seek::<F>),
        };
        // Safety: Just FFI
        let code = unsafe {
            zstd_sys::ZSTD_seekable_initAdvanced(self.0.as_ptr(), custom_file)
        };

        if crate::is_error(code) {
            return Err(code);
        }

        Ok(AdvancedSeekable {
            inner: self,
            src: opaque,
        })
    }
}

/// Seeks the read head to `offset` from `origin`, where origin is either `SEEK_SET`
/// (beginning of file), `SEEK_CUR` (current position) or `SEEK_END` (end of file),
/// as defined in `stdio.h`.
///
/// Returns a non-negative value in case of success, and a negative value in case of failure.
#[cfg(feature = "std")]
unsafe extern "C" fn advanced_seek<S: std::io::Seek>(
    opaque: *mut core::ffi::c_void,
    offset: ::core::ffi::c_longlong,
    origin: ::core::ffi::c_int,
) -> ::core::ffi::c_int {
    use core::convert::TryFrom;
    use std::io::SeekFrom;

    // as defined in stdio.h
    const SEEK_SET: i32 = 0;
    const SEEK_CUR: i32 = 1;
    const SEEK_END: i32 = 2;

    // Safety: The trait boundaries in `init_advanced()` ensure that `opaque` points to an S
    let seeker: &mut S = std::mem::transmute(opaque);
    let pos = match origin {
        SEEK_SET => {
            let Ok(offset) = u64::try_from(offset) else {
                return -1;
            };
            SeekFrom::Start(offset)
        }
        SEEK_CUR => SeekFrom::Current(offset),
        SEEK_END => SeekFrom::End(offset),
        // not possible
        _ => return -1,
    };

    if seeker.seek(pos).is_err() {
        return -1;
    }

    0
}

/// Reads exactly `n` bytes into `buffer`.
///
/// Returns a non-negative value in case of success, and a negative value in case of failure.
#[cfg(feature = "std")]
unsafe extern "C" fn advanced_read<R: std::io::Read>(
    opaque: *mut core::ffi::c_void,
    buffer: *mut core::ffi::c_void,
    n: usize,
) -> ::core::ffi::c_int {
    // Safety: The trait boundaries in `init_advanced()` ensure that `opaque` points to a R
    let reader: &mut R = std::mem::transmute(opaque);
    // Safety: zstd ensures the buffer is allocated and safe to use
    let mut buf = std::slice::from_raw_parts_mut(buffer as *mut u8, n);
    if reader.read_exact(&mut buf).is_err() {
        return -1;
    }

    0
}

/// Indicates that the seek table could not be created.
#[derive(Debug, PartialEq)]
pub struct SeekTableCreateError;

impl core::fmt::Display for SeekTableCreateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("Zstd returned null pointer when creating new seektable from seekable")
    }
}

pub struct SeekTable(NonNull<zstd_sys::ZSTD_seekTable>);

unsafe impl Send for SeekTable {}
unsafe impl Sync for SeekTable {}

impl SeekTable {
    /// Try to create a `SeekTable` from a `Seekable`.
    ///
    /// Memory constrained use cases that manage multiple archives benefit from retaining
    /// multiple archive seek tables without retaining a `Seekable` instance for each.
    pub fn try_from_seekable<'a>(
        value: &Seekable<'a>,
    ) -> Result<Self, SeekTableCreateError> {
        // Safety: Just FFI
        let ptr = unsafe {
            zstd_sys::ZSTD_seekTable_create_fromSeekable(value.0.as_ptr())
        };
        let ptr = NonNull::new(ptr).ok_or(SeekTableCreateError)?;

        Ok(Self(ptr))
    }

    /// Get the number of frames of the underlying seekable object.
    pub fn num_frames(&self) -> u32 {
        // Safety: Just FFI
        unsafe { zstd_sys::ZSTD_seekTable_getNumFrames(self.0.as_ptr()) }
    }

    /// Get the offset of the compressed frame.
    ///
    /// Returns an error if `frame_index` is out of range.
    pub fn frame_compressed_offset(
        &self,
        frame_index: u32,
    ) -> Result<u64, FrameIndexTooLargeError> {
        // Safety: Just FFI
        let offset = unsafe {
            zstd_sys::ZSTD_seekTable_getFrameCompressedOffset(
                self.0.as_ptr(),
                frame_index,
            )
        };

        if offset == SEEKABLE_FRAMEINDEX_TOOLARGE {
            return Err(FrameIndexTooLargeError);
        }

        Ok(offset)
    }

    /// Get the offset of the decompressed frame.
    ///
    /// Returns an error if `frame_index` is out of range.
    pub fn frame_decompressed_offset(
        &self,
        frame_index: u32,
    ) -> Result<u64, FrameIndexTooLargeError> {
        // Safety: Just FFI
        let offset = unsafe {
            zstd_sys::ZSTD_seekTable_getFrameDecompressedOffset(
                self.0.as_ptr(),
                frame_index,
            )
        };

        if offset == SEEKABLE_FRAMEINDEX_TOOLARGE {
            return Err(FrameIndexTooLargeError);
        }

        Ok(offset)
    }

    /// Get the size of the compressed frame.
    ///
    /// Returns an error if `frame_index` is out of range.
    pub fn frame_compressed_size(&self, frame_index: u32) -> SafeResult {
        // Safety: Just FFI
        let code = unsafe {
            zstd_sys::ZSTD_seekTable_getFrameCompressedSize(
                self.0.as_ptr(),
                frame_index,
            )
        };

        parse_code(code)
    }

    /// Get the size of the decompressed frame.
    ///
    /// Returns an error if `frame_index` is out of range.
    pub fn frame_decompressed_size(&self, frame_index: u32) -> SafeResult {
        // Safety: Just FFI
        let code = unsafe {
            zstd_sys::ZSTD_seekTable_getFrameDecompressedSize(
                self.0.as_ptr(),
                frame_index,
            )
        };

        parse_code(code)
    }

    /// Get the frame at the given offset.
    pub fn offset_to_frame_index(&self, offset: u64) -> u32 {
        // Safety: Just FFI
        unsafe {
            zstd_sys::ZSTD_seekTable_offsetToFrameIndex(
                self.0.as_ptr(),
                offset,
            )
        }
    }
}

impl Drop for SeekTable {
    fn drop(&mut self) {
        // Safety: Just FFI
        unsafe {
            zstd_sys::ZSTD_seekTable_free(self.0.as_ptr());
        }
    }
}
