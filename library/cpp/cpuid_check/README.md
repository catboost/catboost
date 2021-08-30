Simple utility to check base target x86 SIMD exensions at startup.

Program may be built with some SIMD extension enabled (e.g. `-msse4.2`). `PEERDIR` to this library adds statrup check that machine where the program is running supports SIMD extension the program is built for.

Currently supported check are: sse4.2, pclmul, aes, avx, avx2 and fma. 

**Note:** the library depends on `util`. 
**Note:** the library adds stratup code and so if `PEERDIR`-ed from `LIBRARY` will do so for all `PROGRAM`-s that (transitively) use the `LIBRARY`. Don't do this!

You normally don't need to `PEERDIR` this library at all. Since making sse4 in Arcadia default this library is used implicitly. It is `PEERDIR`-ed from all `PROGRAM`-s and derived modules (e.g. `PY2_PROGRAM`, but not `GO_PROGRAM` or `JAVA_PROGRAM`).
It is also not applied to `PROGRAM`-s where `NO_UTIL()`, `NO_PLATFORM()` or `ALLOCATOR(FAKE)` set to avoid undesired dependencied. To disable this implicit check use `NO_CPU_CHECK()` macro or `-DCPU_CHECK=no` ya make flag.


