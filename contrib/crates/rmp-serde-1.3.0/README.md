# MessagePack + Serde

This crate connects Rust MessagePack library with [`serde`][serde] providing an ability to
easily serialize and deserialize both Rust built-in types, the standard library and custom data
structures.

## Motivating example

```rust
let buf = rmp_serde::to_vec(&(42, "the Answer")).unwrap();

assert_eq!(
    vec![0x92, 0x2a, 0xaa, 0x74, 0x68, 0x65, 0x20, 0x41, 0x6e, 0x73, 0x77, 0x65, 0x72],
    buf
);

assert_eq!((42, "the Answer"), rmp_serde::from_slice(&buf).unwrap());
```

## Type-based Serialization and Deserialization

Serde provides a mechanism for low boilerplate serialization & deserialization of values to and
from MessagePack via the serialization API.

To be able to serialize a piece of data, it must implement the `serde::Serialize` trait. To be
able to deserialize a piece of data, it must implement the `serde::Deserialize` trait. Serde
provides an annotation to automatically generate the code for these
traits: `#[derive(Serialize, Deserialize)]`.

## Examples

```rust
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use rmp_serde::{Deserializer, Serializer};

#[derive(Debug, PartialEq, Deserialize, Serialize)]
struct Human {
    age: u32,
    name: String,
}

fn main() {
    let mut buf = Vec::new();
    let val = Human {
        age: 42,
        name: "John".into(),
    };

    val.serialize(&mut Serializer::new(&mut buf)).unwrap();
}
```

## Efficient storage of `&[u8]` types

MessagePack can efficiently store binary data. However, Serde's standard derived implementations *do not* use binary representations by default. Serde prefers to represent types like `&[u8; N]` or `Vec<u8>` as arrays of objects of arbitrary/unknown type, and not as slices of bytes. This creates about a 50% overhead in storage size.

Wrap your data in [`serde_bytes`](https://lib.rs/crates/serde_bytes) to store blobs quickly and efficiently. Alternatively, [configure an override in `rmp_serde` to force use of byte slices](https://docs.rs/rmp-serde/latest/rmp_serde/encode/struct.Serializer.html#method.with_bytes).

[serde]: https://serde.rs/
