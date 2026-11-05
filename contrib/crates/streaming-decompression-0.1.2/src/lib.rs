#![forbid(unsafe_code)]
#![doc = include_str!("lib.md")]

pub use fallible_streaming_iterator;
pub use fallible_streaming_iterator::FallibleStreamingIterator;

/// Trait denoting a compressed item. Use `is_compressed` to declare if the item is
/// compressed at runtime (in which case the internal buffer is swapped)
pub trait Compressed {
    #[inline]
    fn is_compressed(&self) -> bool {
        true
    }
}

/// Trait denoting an uncompressed item. Use `buffer_mut` to expose a mutable reference to its
/// internal buffer, which [`Decompressor`] will use to recover a decompressed buffer for re-use.
pub trait Decompressed {
    fn buffer_mut(&mut self) -> &mut Vec<u8>;
}

/// A [`FallibleStreamingIterator`] that decompresses items of type `I` into type `O` via an
/// internal [`Vec<u8>`] that is re-used across items.
/// The purpose of this streaming iterator is to be able to decompress parts of an item `I` into `O`.
pub struct Decompressor<I, O, F, E, II>
where
    I: Compressed,
    O: Decompressed,
    E: std::error::Error,
    II: Iterator<Item = Result<I, E>>,
    F: Fn(I, &mut Vec<u8>) -> Result<O, E>,
{
    iter: II,
    f: F,
    buffer: Vec<u8>,
    current: Option<O>,
    was_decompressed: bool,
}

impl<I, O, F, E, II> Decompressor<I, O, F, E, II>
where
    I: Compressed,
    O: Decompressed,
    E: std::error::Error,
    II: Iterator<Item = Result<I, E>>,
    F: Fn(I, &mut Vec<u8>) -> Result<O, E>,
{
    /// Returns a new [`Decompressor`].
    #[inline]
    pub fn new(iter: II, buffer: Vec<u8>, f: F) -> Self {
        Self {
            iter,
            f,
            buffer,
            current: None,
            was_decompressed: false,
        }
    }

    /// Returns its internal buffer, consuming itself.
    #[inline]
    pub fn into_inner(mut self) -> Vec<u8> {
        self.buffer.clear(); // not leak information
        self.buffer
    }
}

impl<I, O, F, E, II> FallibleStreamingIterator for Decompressor<I, O, F, E, II>
where
    I: Compressed,
    O: Decompressed,
    E: std::error::Error,
    II: Iterator<Item = Result<I, E>>,
    F: Fn(I, &mut Vec<u8>) -> Result<O, E>,
{
    type Item = O;
    type Error = E;

    #[inline]
    fn advance(&mut self) -> Result<(), E> {
        if let Some(page) = self.current.as_mut() {
            if self.was_decompressed {
                self.buffer = std::mem::take(page.buffer_mut());
            }
        }

        let next = self
            .iter
            .next()
            .map(|maybe_page| {
                maybe_page.and_then(|page| {
                    self.was_decompressed = page.is_compressed();
                    (self.f)(page, &mut self.buffer)
                })
            })
            .transpose()?;
        self.current = next;
        Ok(())
    }

    #[inline]
    fn get(&self) -> Option<&Self::Item> {
        self.current.as_ref()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct CompressedItem {
        pub metadata: String,
        pub data: Vec<u8>,
    }
    impl Compressed for CompressedItem {
        fn is_compressed(&self) -> bool {
            self.metadata == "is_compressed"
        }
    }

    #[derive(Debug, PartialEq)]
    struct DecompressedItem {
        pub metadata: String,
        pub data: Vec<u8>,
    }

    impl Decompressed for DecompressedItem {
        fn buffer_mut(&mut self) -> &mut Vec<u8> {
            &mut self.data
        }
    }

    fn decompress(
        mut i: CompressedItem,
        buffer: &mut Vec<u8>,
    ) -> Result<DecompressedItem, std::convert::Infallible> {
        if i.is_compressed() {
            // the actual decompression, more complex stuff can happen.
            buffer.clear();
            buffer.extend(&mut i.data.iter());
        } else {
            std::mem::swap(&mut i.data, buffer);
        };
        Ok(DecompressedItem {
            metadata: i.metadata,
            data: std::mem::take(buffer),
        })
    }

    #[test]
    fn test_basics_uncompressed() {
        let item = CompressedItem {
            metadata: "not_compressed".to_string(),
            data: vec![1, 2, 3],
        };
        let iter = vec![Ok(item)].into_iter();

        let buffer = vec![1];
        let mut decompressor = Decompressor::new(iter, buffer, decompress);

        let item = decompressor.next().unwrap().unwrap();
        assert_eq!(item.data, vec![1, 2, 3]);
        assert_eq!(item.metadata, "not_compressed".to_string());
        assert_eq!(decompressor.next().unwrap(), None);

        // i.e. the internal buffer was not used.
        assert_eq!(decompressor.into_inner().capacity(), 0);
    }

    #[test]
    fn test_basics_compressed() {
        let item = CompressedItem {
            metadata: "is_compressed".to_string(),
            data: vec![1, 2, 3],
        };
        let iter = vec![Ok(item)].into_iter();

        let buffer = vec![1, 2];
        let mut decompressor = Decompressor::new(iter, buffer, decompress);

        let item = decompressor.next().unwrap().unwrap();
        assert_eq!(item.data, vec![1, 2, 3]);
        assert_eq!(item.metadata, "is_compressed".to_string());
        assert_eq!(decompressor.next().unwrap(), None);

        // i.e. after the last `next`, the last item is consumed and the internal buffer
        // contains its data
        assert!(decompressor.into_inner().capacity() > 0);
    }
}
