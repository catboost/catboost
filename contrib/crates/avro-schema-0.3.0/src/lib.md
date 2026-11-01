Welcome to avro-schema's documentation. Thanks for checking it out!

This is a library containing declarations of the
[Avro specification](https://avro.apache.org/docs/current/spec.html)
in Rust's struct and enums together with serialization and deserialization
implementations based on `serde_json`.

It also contains basic functionality to read and deserialize Avro's file's metadata
and blocks.

Example of reading a file:

```rust
use std::convert::TryInto;
use std::fs::File;
use std::io::BufReader;

use avro_schema::error::Error;
use avro_schema::read::fallible_streaming_iterator::FallibleStreamingIterator;

fn read_avro(path: &str) -> Result<(), Error> {
    let file = &mut BufReader::new(File::open(path)?);

    let metadata = avro_schema::read::read_metadata(file)?;

    println!("{:#?}", metadata);

    let mut blocks =
        avro_schema::read::BlockStreamingIterator::new(file, metadata.compression, metadata.marker);

    while let Some(block) = blocks.next()? {
        let _fields = &metadata.record.fields;
        let length = block.number_of_rows;
        let mut block: &[u8] = block.data.as_ref();
        // at this point you can deserialize the block based on `_fields` according
        // to avro's specification. Note that `Block` is already decompressed.
        // for example, if there was a single field with f32, we would use
        for _ in 0..length {
            let (item, remaining) = block.split_at(4);
            block = remaining;
            let _value = f32::from_le_bytes(item.try_into().unwrap());
            // if there were more fields, we would need to consume (or skip) the remaining
            // here. You can use `avro_schema::read::decode::zigzag_i64` for integers :D
        }
    }

    Ok(())
}
```

Example of writing a file

```rust
use std::fs::File;

use avro_schema::error::Error;
use avro_schema::file::Block;
use avro_schema::schema::{Field, Record, Schema};

fn write_avro(compression: Option<avro_schema::file::Compression>) -> Result<(), Error> {
    let mut file = File::create("test.avro")?;

    let record = Record::new("", vec![Field::new("value", Schema::Float)]);

    avro_schema::write::write_metadata(&mut file, record, compression)?;

    // given some data:
    let array = vec![1.0f32, 2.0];

    // we need to create a `Block`
    let mut data: Vec<u8> = vec![];
    for item in array.iter() {
        let bytes = item.to_le_bytes();
        data.extend(bytes);
    }
    let mut block = Block::new(array.len(), data);

    // once completed, we compress it
    let mut compressed_block = avro_schema::file::CompressedBlock::default();
    let _ = avro_schema::write::compress(&mut block, &mut compressed_block, compression)?;

    // and finally write it to the file
    avro_schema::write::write_block(&mut file, &compressed_block)?;

    Ok(())
}
```
