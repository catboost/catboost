extern crate std;
use crate as zstd_safe;

use self::std::vec::Vec;

const INPUT: &[u8] = b"Rust is a multi-paradigm system programming language focused on safety, especially safe concurrency. Rust is syntactically similar to C++, but is designed to provide better memory safety while maintaining high performance.";
const LONG_CONTENT: &str = include_str!("lib.rs");

#[cfg(feature = "std")]
#[test]
fn test_writebuf() {
    use zstd_safe::WriteBuf;

    let mut data = Vec::with_capacity(10);
    unsafe {
        data.write_from(|ptr, n| {
            assert!(n >= 4);
            let ptr = ptr as *mut u8;
            ptr.write(0);
            ptr.add(1).write(1);
            ptr.add(2).write(2);
            ptr.add(3).write(3);
            Ok(4)
        })
    }
    .unwrap();
    assert_eq!(data.as_slice(), &[0, 1, 2, 3]);

    let mut cursor = std::io::Cursor::new(&mut data);
    // Here we use a position larger than the actual data.
    // So expect the data to be zero-filled.
    cursor.set_position(6);
    unsafe {
        cursor.write_from(|ptr, n| {
            assert!(n >= 4);
            let ptr = ptr as *mut u8;
            ptr.write(4);
            ptr.add(1).write(5);
            ptr.add(2).write(6);
            ptr.add(3).write(7);
            Ok(4)
        })
    }
    .unwrap();

    assert_eq!(data.as_slice(), &[0, 1, 2, 3, 0, 0, 4, 5, 6, 7]);
}

#[cfg(feature = "std")]
#[test]
fn test_simple_cycle() {
    let mut buffer = std::vec![0u8; 256];
    let written = zstd_safe::compress(&mut buffer, INPUT, 3).unwrap();
    let compressed = &buffer[..written];

    let mut buffer = std::vec![0u8; 256];
    let written = zstd_safe::decompress(&mut buffer, compressed).unwrap();
    let decompressed = &buffer[..written];

    assert_eq!(INPUT, decompressed);
}

#[test]
fn test_cctx_cycle() {
    let mut buffer = std::vec![0u8; 256];
    let mut cctx = zstd_safe::CCtx::default();
    let written = cctx.compress(&mut buffer[..], INPUT, 1).unwrap();
    let compressed = &buffer[..written];

    let mut dctx = zstd_safe::DCtx::default();
    let mut buffer = std::vec![0u8; 256];
    let written = dctx.decompress(&mut buffer[..], compressed).unwrap();
    let decompressed = &buffer[..written];

    assert_eq!(INPUT, decompressed);
}

#[test]
fn test_dictionary() {
    // Prepare some content to train the dictionary.
    let bytes = LONG_CONTENT.as_bytes();
    let line_sizes: Vec<usize> =
        LONG_CONTENT.lines().map(|line| line.len() + 1).collect();

    // Train the dictionary
    let mut dict_buffer = std::vec![0u8; 100_000];
    let written =
        zstd_safe::train_from_buffer(&mut dict_buffer[..], bytes, &line_sizes)
            .unwrap();
    let dict_buffer = &dict_buffer[..written];

    // Create pre-hashed dictionaries for (de)compression
    let cdict = zstd_safe::create_cdict(dict_buffer, 3);
    let ddict = zstd_safe::create_ddict(dict_buffer);

    // Compress data
    let mut cctx = zstd_safe::CCtx::default();
    cctx.ref_cdict(&cdict).unwrap();

    let mut buffer = std::vec![0u8; 1024 * 1024];
    // First, try to compress without a dict
    let big_written = zstd_safe::compress(&mut buffer[..], bytes, 3).unwrap();

    let written = cctx
        .compress2(&mut buffer[..], bytes)
        .map_err(zstd_safe::get_error_name)
        .unwrap();

    assert!(big_written > written);
    let compressed = &buffer[..written];

    // Decompress data
    let mut dctx = zstd_safe::DCtx::default();
    dctx.ref_ddict(&ddict).unwrap();

    let mut buffer = std::vec![0u8; 1024 * 1024];
    let written = dctx
        .decompress(&mut buffer[..], compressed)
        .map_err(zstd_safe::get_error_name)
        .unwrap();
    let decompressed = &buffer[..written];

    // Profit!
    assert_eq!(bytes, decompressed);
}

#[test]
fn test_checksum() {
    let mut buffer = std::vec![0u8; 256];
    let mut cctx = zstd_safe::CCtx::default();
    cctx.set_parameter(zstd_safe::CParameter::ChecksumFlag(true))
        .unwrap();
    let written = cctx.compress2(&mut buffer[..], INPUT).unwrap();
    let compressed = &mut buffer[..written];

    let mut dctx = zstd_safe::DCtx::default();
    let mut buffer = std::vec![0u8; 1024*1024];
    let written = dctx
        .decompress(&mut buffer[..], compressed)
        .map_err(zstd_safe::get_error_name)
        .unwrap();
    let decompressed = &buffer[..written];

    assert_eq!(INPUT, decompressed);

    // Now try again with some corruption
    // TODO: Find a mutation that _wouldn't_ be detected without checksums.
    // (Most naive changes already trigger a "corrupt block" error.)
    if let Some(last) = compressed.last_mut() {
        *last = last.saturating_sub(1);
    }
    let err = dctx
        .decompress(&mut buffer[..], compressed)
        .map_err(zstd_safe::get_error_name)
        .err()
        .unwrap();
    // The error message will complain about the checksum.
    assert!(err.contains("checksum"));
}

#[cfg(all(feature = "experimental", feature = "std"))]
#[test]
fn test_upper_bound() {
    let mut buffer = std::vec![0u8; 256];

    assert!(zstd_safe::decompress_bound(&buffer).is_err());

    let written = zstd_safe::compress(&mut buffer, INPUT, 3).unwrap();
    let compressed = &buffer[..written];

    assert_eq!(
        zstd_safe::decompress_bound(&compressed),
        Ok(INPUT.len() as u64)
    );
}

#[cfg(feature = "seekable")]
#[test]
fn test_seekable_cycle() {
    let seekable_archive = new_seekable_archive(INPUT);
    let mut seekable = crate::seekable::Seekable::create();
    seekable
        .init_buff(&seekable_archive)
        .map_err(zstd_safe::get_error_name)
        .unwrap();

    decompress_seekable(&mut seekable);

    // Check that the archive can also be decompressed by a regular function
    let mut buffer = std::vec![0u8; 256];
    let written = zstd_safe::decompress(&mut buffer[..], &seekable_archive)
        .map_err(zstd_safe::get_error_name)
        .unwrap();
    let decompressed = &buffer[..written];
    assert_eq!(INPUT, decompressed);

    // Trigger FrameIndexTooLargeError
    let frame_index = seekable.num_frames() + 1;
    assert_eq!(
        seekable.frame_compressed_offset(frame_index).unwrap_err(),
        crate::seekable::FrameIndexTooLargeError
    );
}

#[cfg(feature = "seekable")]
#[test]
fn test_seekable_seek_table() {
    use crate::seekable::{FrameIndexTooLargeError, SeekTable, Seekable};

    let seekable_archive = new_seekable_archive(INPUT);
    let mut seekable = Seekable::create();

    // Assert that creating a SeekTable from an uninitialized seekable errors.
    // This led to segfaults with zstd versions prior v1.5.7
    assert!(SeekTable::try_from_seekable(&seekable).is_err());

    seekable
        .init_buff(&seekable_archive)
        .map_err(zstd_safe::get_error_name)
        .unwrap();

    // Try to create a seek table from the seekable
    let seek_table =
        { SeekTable::try_from_seekable(&seekable).unwrap() };

    // Seekable and seek table should return the same results
    assert_eq!(seekable.num_frames(), seek_table.num_frames());
    assert_eq!(
        seekable.frame_compressed_offset(2).unwrap(),
        seek_table.frame_compressed_offset(2).unwrap()
    );
    assert_eq!(
        seekable.frame_decompressed_offset(2).unwrap(),
        seek_table.frame_decompressed_offset(2).unwrap()
    );
    assert_eq!(
        seekable.frame_compressed_size(2).unwrap(),
        seek_table.frame_compressed_size(2).unwrap()
    );
    assert_eq!(
        seekable.frame_decompressed_size(2).unwrap(),
        seek_table.frame_decompressed_size(2).unwrap()
    );

    // Trigger FrameIndexTooLargeError
    let frame_index = seekable.num_frames() + 1;
    assert_eq!(
        seek_table.frame_compressed_offset(frame_index).unwrap_err(),
        FrameIndexTooLargeError
    );
}

#[cfg(all(feature = "std", feature = "seekable"))]
#[test]
fn test_seekable_advanced_cycle() {
    use crate::seekable::Seekable;
    use std::{boxed::Box, io::Cursor};

    // Wrap the archive in a cursor that implements Read and Seek,
    // a file would also work
    let seekable_archive = Cursor::new(new_seekable_archive(INPUT));
    let mut seekable = Seekable::create()
        .init_advanced(Box::new(seekable_archive))
        .map_err(zstd_safe::get_error_name)
        .unwrap();

    decompress_seekable(&mut seekable);
}

#[cfg(feature = "seekable")]
fn new_seekable_archive(input: &[u8]) -> Vec<u8> {
    use crate::{seekable::SeekableCStream, InBuffer, OutBuffer};

    // Make sure the buffer is big enough
    // The buffer needs to be bigger as the uncompressed data here as the seekable archive has
    // more meta data than actual compressed data because the input is really small and we use
    // a max_frame_size of 64, which is way to small for real-world usages.
    let mut buffer = std::vec![0u8; 512];
    let mut cstream = SeekableCStream::create();
    cstream
        .init(3, true, 64)
        .map_err(zstd_safe::get_error_name)
        .unwrap();
    let mut in_buffer = InBuffer::around(input);
    let mut out_buffer = OutBuffer::around(&mut buffer[..]);

    // This could get stuck if the buffer is too small
    while in_buffer.pos() < in_buffer.src.len() {
        cstream
            .compress_stream(&mut out_buffer, &mut in_buffer)
            .map_err(zstd_safe::get_error_name)
            .unwrap();
    }

    // Make sure everything is flushed to out_buffer
    loop {
        if cstream
            .end_stream(&mut out_buffer)
            .map_err(zstd_safe::get_error_name)
            .unwrap()
            == 0
        {
            break;
        }
    }

    Vec::from(out_buffer.as_slice())
}

#[cfg(feature = "seekable")]
fn decompress_seekable(seekable: &mut crate::seekable::Seekable<'_>) {
    // Make the buffer as big as max_frame_size so it can hold a complete frame
    let mut buffer = std::vec![0u8; 64];
    // Decompress only the first frame
    let written = seekable
        .decompress(&mut buffer[..], 0)
        .map_err(zstd_safe::get_error_name)
        .unwrap();
    let decompressed = &buffer[..written];
    assert!(INPUT.starts_with(decompressed));
    assert_eq!(decompressed.len(), 64);

    // Make the buffer big enough to hold the complete input
    let mut buffer = std::vec![0u8; 256];
    // Decompress everything
    let written = seekable
        .decompress(&mut buffer[..], 0)
        .map_err(zstd_safe::get_error_name)
        .unwrap();
    let decompressed = &buffer[..written];
    assert_eq!(INPUT, decompressed);
}
