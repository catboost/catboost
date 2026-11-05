#[cfg(feature = "no_std")]
use alloc::vec::Vec;
#[cfg(feature = "no_std")]
use core::cmp;
#[cfg(feature = "no_std")]
use hashbrown::HashMap;
#[cfg(not(feature = "no_std"))]
use std::{cmp, collections::HashMap};

use super::Code;
use super::Lz77Encode;
use super::Sink;

/// A [`Lz77Encode`] implementation used by default.
#[derive(Debug)]
pub struct DefaultLz77Encoder {
    window_size: u16,
    max_length: u16,
    buf: Vec<u8>,
}

impl DefaultLz77Encoder {
    /// Makes a new encoder instance.
    ///
    /// # Examples
    /// ```
    /// use libflate::deflate;
    /// use libflate::lz77::{self, Lz77Encode, DefaultLz77Encoder};
    ///
    /// let lz77 = DefaultLz77Encoder::new();
    /// assert_eq!(lz77.window_size(), lz77::MAX_WINDOW_SIZE);
    ///
    /// let options = deflate::EncodeOptions::with_lz77(lz77);
    /// let _deflate = deflate::Encoder::with_options(Vec::new(), options);
    /// ```
    pub fn new() -> Self {
        DefaultLz77EncoderBuilder::new().build()
    }

    /// Makes a new encoder instance with specified window size.
    ///
    /// Larger window size is prefered to raise compression ratio,
    /// but it may require more working memory to encode and decode data.
    ///
    /// # Examples
    /// ```
    /// use libflate::deflate;
    /// use libflate::lz77::{self, Lz77Encode, DefaultLz77Encoder};
    ///
    /// let lz77 = DefaultLz77Encoder::with_window_size(1024);
    /// assert_eq!(lz77.window_size(), 1024);
    ///
    /// let options = deflate::EncodeOptions::with_lz77(lz77);
    /// let _deflate = deflate::Encoder::with_options(Vec::new(), options);
    /// ```
    pub fn with_window_size(size: u16) -> Self {
        DefaultLz77EncoderBuilder::new()
            .window_size(cmp::min(size, super::MAX_WINDOW_SIZE))
            .build()
    }
}

impl Default for DefaultLz77Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Lz77Encode for DefaultLz77Encoder {
    fn encode<S>(&mut self, buf: &[u8], sink: S)
    where
        S: Sink,
    {
        self.buf.extend_from_slice(buf);
        if self.buf.len() >= self.window_size as usize * 8 {
            self.flush(sink);
        }
    }
    fn flush<S>(&mut self, mut sink: S)
    where
        S: Sink,
    {
        let mut prefix_table = PrefixTable::new(self.buf.len());
        let mut i = 0;
        let end = cmp::max(3, self.buf.len()) - 3;
        while i < end {
            let key = prefix(&self.buf[i..]);
            let matched = prefix_table.insert(key, i as u32);
            if let Some(j) = matched.map(|j| j as usize) {
                let distance = i - j;
                if distance <= self.window_size as usize {
                    let length = 3 + longest_common_prefix(
                        &self.buf,
                        i + 3,
                        j + 3,
                        self.max_length as usize,
                    );
                    sink.consume(Code::Pointer {
                        length,
                        backward_distance: distance as u16,
                    });
                    for k in (i..).take(length as usize).skip(1) {
                        if k >= end {
                            break;
                        }
                        prefix_table.insert(prefix(&self.buf[k..]), k as u32);
                    }
                    i += length as usize;
                    continue;
                }
            }
            sink.consume(Code::Literal(self.buf[i]));
            i += 1;
        }
        for b in &self.buf[i..] {
            sink.consume(Code::Literal(*b));
        }
        self.buf.clear();
    }
    fn window_size(&self) -> u16 {
        self.window_size
    }
}

#[inline]
fn prefix(input_buf: &[u8]) -> [u8; 3] {
    let buf: &[u8] = &input_buf[..3]; // perform bounds check once
    [buf[0], buf[1], buf[2]]
}

#[inline]
fn longest_common_prefix(buf: &[u8], i: usize, j: usize, max: usize) -> u16 {
    buf[i..]
        .iter()
        .take(max - 3)
        .zip(&buf[j..])
        .take_while(|&(x, y)| x == y)
        .count() as u16
}

#[derive(Debug)]
enum PrefixTable {
    Small(HashMap<[u8; 3], u32>),
    Large(LargePrefixTable),
}
impl PrefixTable {
    fn new(bytes: usize) -> Self {
        if bytes < super::MAX_WINDOW_SIZE as usize {
            PrefixTable::Small(HashMap::new())
        } else {
            PrefixTable::Large(LargePrefixTable::new())
        }
    }

    #[inline]
    fn insert(&mut self, prefix: [u8; 3], position: u32) -> Option<u32> {
        match *self {
            PrefixTable::Small(ref mut x) => x.insert(prefix, position),
            PrefixTable::Large(ref mut x) => x.insert(prefix, position),
        }
    }
}

#[derive(Debug)]
struct LargePrefixTable {
    table: Vec<Vec<(u8, u32)>>,
}
impl LargePrefixTable {
    fn new() -> Self {
        LargePrefixTable {
            table: (0..=0xFFFF).map(|_| Vec::new()).collect(),
        }
    }

    #[inline]
    fn insert(&mut self, prefix: [u8; 3], position: u32) -> Option<u32> {
        let p0 = prefix[0] as usize;
        let p1 = prefix[1] as usize;
        let p2 = prefix[2];

        let i = (p0 << 8) + p1;
        let positions = &mut self.table[i];
        for &mut (key, ref mut value) in positions.iter_mut() {
            if key == p2 {
                let old = *value;
                *value = position;
                return Some(old);
            }
        }
        positions.push((p2, position));
        None
    }
}

/// Type for constructing instances of [`DefaultLz77Encoder`].
///
/// # Examples
/// ```
/// use libflate_lz77::{
///     DefaultLz77EncoderBuilder,
///     MAX_LENGTH,
///     MAX_WINDOW_SIZE,
/// };
///
/// // Produce an encoder explicitly with the default window size and max copy length
/// let _encoder = DefaultLz77EncoderBuilder::new()
///     .window_size(MAX_WINDOW_SIZE)
///     .max_length(MAX_LENGTH)
///     .build();
/// ```
#[derive(Debug)]
pub struct DefaultLz77EncoderBuilder {
    window_size: u16,
    max_length: u16,
}

impl DefaultLz77EncoderBuilder {
    /// Create a builder with the default parameters for the encoder.
    pub fn new() -> Self {
        DefaultLz77EncoderBuilder {
            window_size: super::MAX_WINDOW_SIZE,
            max_length: super::MAX_LENGTH,
        }
    }

    /// Set the size of the sliding search window used during compression.
    ///
    /// Larger values require more memory. The standard window size may be
    /// unsuitable for a particular Sink; for example, if the encoding used
    /// cannot express pointer distances past a certain size, you would want the
    /// window size to be no greater than the Sink's limit.
    pub fn window_size(self, window_size: u16) -> Self {
        DefaultLz77EncoderBuilder {
            window_size: cmp::min(window_size, super::MAX_WINDOW_SIZE),
            ..self
        }
    }

    /// Set the maximum length of a pointer command this encoder will emit.
    ///
    /// Some uses of LZ77 may not be able to encode pointers of the standard
    /// maximum length of 258 bytes. In this case, you may set your own maximum
    /// which can be encoded by the Sink.
    pub fn max_length(self, max_length: u16) -> Self {
        DefaultLz77EncoderBuilder {
            max_length: cmp::min(max_length, super::MAX_LENGTH),
            ..self
        }
    }

    /// Build the encoder with the builder state's parameters.
    pub fn build(self) -> DefaultLz77Encoder {
        DefaultLz77Encoder {
            window_size: self.window_size,
            max_length: self.max_length,
            buf: Vec::new(),
        }
    }
}

impl Default for DefaultLz77EncoderBuilder {
    fn default() -> Self {
        Self::new()
    }
}
