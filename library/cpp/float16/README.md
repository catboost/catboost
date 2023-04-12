### Float16 - half-precision floating point ([binary16](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) from IEEE 754-2008)

This library implements simple float16 ⇔ float32 conversations that do not require special processor instructions.

Sequence functions:
- `UnpackFloat16SequenceAuto()`: float16 → float32 conversion for sequences
- `PackFloat16SequenceAuto()`: float32 → float16 conversion for sequences

For both of the functions above, `{Pack,Unpack}Float16SequenceIntrisincs()` will be called if F16C flag is set for CPU. Otherwise, a fallback function will be used.

SEARCH TAGS: float16, half precision floating point, fp16
