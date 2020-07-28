gulin@
A fast implementation of exp() function. Comparing to libc implementation:
1. This one is less accurate
2. No Inf/NaNs handling
3. Larger table. This results that if you have 3 calls to exp(), then this one is even slower. Speedup achieved only if the table is well cached. But if you have a slow exp it means that you call it not 3 times.
