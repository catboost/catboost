This lightweight module can be used by anyone interested in YTAlloc
functionality (memory allocation and disposal, memory tagging, etc).

If YTAlloc happens to be linked in, it provides an efficient implementation.
Otherwise (non-YTAlloc build), weak implementations from fallback.cpp
are used.