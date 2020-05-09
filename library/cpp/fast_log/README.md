cobat@
Fast implementations of logarithm function for base 2 and base e.
- FastLogf() and FastLog2f() - Accuracy: ~1.e-5, Speed: ~3x over logf()
- FasterLogf() and FasterLog2f() - Accuracy: ~1.e-3, Speed: ~6x over logf()
- FastestLogf() and FastestLog2f() -  Accuracy: ~1.e-2, Speed: ~12x over logf()

Input check is supported only in Debug mode and is implemented via Y_ASSERT.
