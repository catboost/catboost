### Float16 - halfe precision floating point format IEEE 754-2008
https://en.wikipedia.org/wiki/Half-precision_floating-point_format

This library implements simple float16 <-> float32 conversations that do not require special processor instructions.

NOTE: this implementation do not try to normalize input data, so if input float32 is subnormal, the result can be zero if it in fact representable in float16

Intrisincs functions:
    UnpackFloat16Sequence float16->float32 conversation for sequences 
        UnpackFloat16SequenceIntrisincs will be called if F16C flag is set for CPU
        UnpackFloat16SequenceAuto is fallback
