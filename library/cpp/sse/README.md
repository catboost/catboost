Overview
===
This library provides the implementation of Intel SSE intrinsics for other CPU architectures. Currently supports PowerPC via translation to AltiVec and ARM via NEON. In some cases, falls back to software emulation if there's no corresponding instruction in the target instruction set.

Usage
===
Include library/cpp/sse/sse.h and use the needed intrinsics. Implementation will be selected based on the target architecture of the used toolchain.
