This is a simple library for block data compression (this means data is compressed/uncompressed
by whole blocks in memory). It's a lite-version of the `library/cpp/codecs`. Lite here means that it
provide only well-known compression algorithms, without the possibility of learning.

There are two possible ways to work with it.

Codec by name
=============
Use `NBlockCodec::Codec` to obtain the codec by name. The codec can be asked to compress
or decompress something and in various ways.

To get a full list of codecs there is a function `NBlockCodecs::ListAllCodecs()`.

Streaming
=========
Use `stream.h` to obtain simple streams over block codecs (buffer data, compress them by blocks,
write to the resulting stream).

Using codec plugins
===================
If you don't want your code to bloat from unused codecs, you can use the small version of the
library: `library/cpp/blockcodecs/core`. In that case, you need to manually set `PEERDIR()`s to
needed codecs (i.e. `PEERDIR(library/cpp/blockcodecs/codecs/lzma)`).
