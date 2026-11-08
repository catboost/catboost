This crate contains a [`FallibleStreamingIterator`] optimized for decompressions.

A typical problem that libraries working with compressed formats face is that they need to preserve
an intermediary buffer to decompress multiple items. Specifically, implementations that use

```rust
fn decompress(compressed: Vec<u8>) -> Vec<u8> {
    unimplemented!("Decompress")
}
```

are ineficient because they will need to allocate a new `Vec<u8>` for every decompression, and this
allocation scales with the average _decompressed_ size the items.

The typical solution for this problem is to offer

```rust
fn decompress(compressed: Vec<u8>, decompressed: &mut Vec<u8>) {
    decompressed.clear();
    unimplemented!("Decompress into `decompressed`, maybe re-allocing it.")
}
```

Such API avoids one allocation per item, but requires the user to deal with preserving `decompressed`
between iterations. Such pattern is not possible to achieve with [`Iterator`] API atm.

This crate offers [`Decompressor`], a [`FallibleStreamingIterator`] that consumes an [`Iterator`] of compressed items
that yields uncompressed items, maintaining an internal [`Vec<u8>`] that is automatically re-used across items.

# Example

```rust
use streaming_codec::{Decompressor, Compressed, Decompressed, FallibleStreamingIterator};

// An item that is decompressable
#[derive(Debug, PartialEq)]
struct CompressedItem {
    pub metadata: String,
    pub data: Vec<u8>,
}
impl Compressed for CompressedItem {
    fn is_compressed(&self) -> bool {
        // whether it is decompressed or not depends on some metadata.
        self.metadata == "is_compressed"
    }
}

// A decompressed item
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

// the decompression function. This could call LZ4, Snappy, etc.
fn decompress(
    mut i: CompressedItem,
    buffer: &mut Vec<u8>,
) -> Result<DecompressedItem, std::convert::Infallible> {
    if i.is_compressed() {
        // the actual decompression, here identity, but more complex stuff can happen.
        buffer.clear();
        buffer.extend(&mut i.data.iter().rev());
    } else {
        std::mem::swap(&mut i.data, buffer);
    };
    Ok(DecompressedItem {
        metadata: i.metadata,
        data: std::mem::take(buffer),
    })
}

fn main() -> Result<(), std::convert::Infallible> {
   // consider some compressed items
   let item1 = CompressedItem {
       metadata: "is_compressed".to_string(),
       data: vec![1, 2, 3],
   };
   let item2 = CompressedItem {
       metadata: "is_compressed".to_string(),
       data: vec![4, 5, 6],
   };
   let iter = vec![Ok(item1), Ok(item2)].into_iter();

   let buffer = vec![0; 4];  // the internal buffer: it could contain anything.
   let mut decompressor = Decompressor::new(iter, buffer, decompress);

   let item = decompressor.next()?.unwrap();
   // the item was decompressed
   assert_eq!(item.data, vec![3, 2, 1]);
   assert_eq!(item.metadata, "is_compressed".to_string());

   let item = decompressor.next()?.unwrap();
   // the item was decompressed
   assert_eq!(item.data, vec![6, 5, 4]);
   assert_eq!(item.metadata, "is_compressed".to_string());

   assert_eq!(decompressor.next()?, None);

   // we can re-use the internal buffer if we wish to
   let internal = decompressor.into_inner();
   assert_eq!(internal, vec![6, 5, 4]);
   Ok(())
}
```

## License

Licensed under either of

 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
