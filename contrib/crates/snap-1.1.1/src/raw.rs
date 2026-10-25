/*!
This module provides a raw Snappy encoder and decoder.

A raw Snappy encoder/decoder can only compress/decompress a fixed amount of
data at a time. For this reason, this module is lower level and more difficult
to use than the higher level streaming readers and writers exposed as part of
the [`read`](../read/index.html) and [`write`](../write/index.html) modules.

Generally, one only needs to use the raw format if some other source is
generating raw Snappy compressed data and you have no choice but to do the
same. Otherwise, the Snappy frame format should probably always be preferred.
*/
pub use crate::compress::{max_compress_len, Encoder};
pub use crate::decompress::{decompress_len, Decoder};
