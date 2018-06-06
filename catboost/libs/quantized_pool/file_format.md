File format was designed to simplify and optimize (by memory) its creation and also make it
mappable.

File with quantized pool will have following structure:

1.  | Magic | -- "CatboostQuantizedPoolPart" (with terminating zero)
2.  | 4-byte for Version |
3.  | 4-byte for Version hash |
4.  | 4-byte for MetaInfoSize |
5.  | padding for 16-byte alignment |
6.  | metainfo of size MetaInfoSize |
7.  --------------------------------------------------------------
    | 4-byte ChunkSize1 | padding for 16-byte alignment | Chunk1 |
    | 4-byte ChunkSize2 | padding for 16-byte alignment | Chunk2 |
    | .......................................................... |
    | 4-byte ChunkSizeN | padding for 16-byte alignment | ChunkN |
    --------------------------------------------------------------
8.  | 4-byte FeatureCount |
9.  ------------------------------------------------------------------------------------
    | 4-byte TrueFeatureIndex | 4-byte ChunkCount                                      |
    | 8-byte ChunkSize1Offset | 4-byte DocumentOffset1 | 4-byte DocumentsInChunk1Count |
    | 8-byte ChunkSize2Offset | 4-byte DocumentOffset2 | 4-byte DocumentsInChunk1Count |
    | ................................................................................ |
    | 8-byte ChunkSizeNOffset | 4-byte DocumentOffsetN | 4-byte DocumentsInChunkNCount |
    ------------------------------------------------------------------------------------
    Repeated FeatureCount times.
10. 8-byte offset of 7
11. 8-byte offset of 8.

NOTE: Offsets in 9 and 10 are given from the beginning of file.
NOTE: All number are LE
