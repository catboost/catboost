File format was designed to simplify and optimize (by memory) its creation and also make it
mappable.

File with quantized pool will have following structure:

```
1.  | Magic | -- "CatboostQuantizedPool" (with terminating zero)
2.  | 4-byte for Version |
3.  | 4-byte for Version hash |
4.  | 4-byte for MetaInfoSize |
5.  | padding for 16-byte alignment |
6.  | metainfo of size MetaInfoSize |
7.  ------------------------------------------
    | padding for 16-byte alignment | Chunk1 |
    | padding for 16-byte alignment | Chunk2 |
    | ...................................... |
    | padding for 16-byte alignment | ChunkN |
    ------------------------------------------
8.  | 4-byte PoolMetainfoSize | PoolMetainfo |
9   | 4-byte QuantizationSchemaSize | QuantizationSchema |
10. | 4-byte NonConstFeatureCount |
11. ----------------------------------------------------------------------------------------------------
    | 4-byte TrueFeatureIndex | 4-byte ChunkCount                                                      |
    | 4-byte ChunkSize1 | 8-byte Chunk1Offset | 4-byte DocumentOffset1 | 4-byte DocumentsInChunk1Count |
    | 4-byte ChunkSize2 | 8-byte Chunk2Offset | 4-byte DocumentOffset2 | 4-byte DocumentsInChunk2Count |
    | ................................................................................................ |
    | 4-byte ChunkSizeN | 8-byte ChunkNOffset | 4-byte DocumentOffsetN | 4-byte DocumentsInChunkNCount |
    ----------------------------------------------------------------------------------------------------
    Repeated FeatureCount times.
12. 8-byte offset of 7
13. 8-byte offset of 8
14. 8-byte offset of 9
15. 8-byte offset of 10
16. | MagicEnd | -- "CatboostQuantizedPoolEnd" (with terminating zero)
```

NOTE: Offsets in 11, 12, 13, 14, and 15 are given from the beginning of file.
NOTE: All number are LE
