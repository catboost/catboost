Library for calculating the distance between vectors
====================================================

Each vector consists of two concatenated vectors.

The distance between two such vectors is considered as the harmonic mean of the cosine of the first internal vectors
and the cosine of the second internal vectors.

A = a1 || a2
B = b1 || b2

|AB| = 2 * cos(a1, b1) * cos(a2, b2)/ ( cos(a1, b1) + cos(a2, b2) )
