This is a simple library for block (this means data is compressed/uncompressed by whole blocks in memory) data compression. It's a lite-version of the library/codecs. Lite here means that it provide only well-known compression algorithms, without the possibility of learning.
Interfaces:
1)
 - NBlockCodec::Codec - returns the codec by name. The codec can be asked to compress or decompress something and in various ways.
 - To get a full list codecs there is a function NBlockCodecs::ListAllCodecs().
2) stream.h - simple streams over block codecs (buffer data, compress them by blocks, write to the resulting stream).
