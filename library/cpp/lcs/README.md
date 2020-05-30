A set of algorithms for approximate string matching.
====================================================

**Lcs_via_lis.h**
Combinatorial algorithm LCS (Longest common subsequence) through LIS (Longest increasing subsequence) for rows S1 and S2 of lengths n and m respectively (we assume that n < m).

Complexity is O(r log n) by time and O(r) of additional memory, where r is the number of pairs (i, j) for which S1[i] = S2[j]. Thus for the uniform distribution of letters of an alphabet of s elements the estimate is O(nm / s * log n).

Effective for large alphabets s = O(n) (like hashes of words in a text). If only the LCS length is needed, the LCS itself will not be built.
See Gusfield, Algorithms on Strings, Trees and Sequences, 1997, Chapt.12.5
Or here: http://www.cs.ucf.edu/courses/cap5937/fall2004/Longest%20common%20subsequence.pdf

#### Summary of the idea:
Let's suppose we have a sequence of numbers.

Denote by:
- IS is a subsequence strictly increasing from left to right.
- LIS is the largest IS of the original sequence.
- DS is a subsequence that does not decrease from left to right.
- C is a covering of disjoint DS of the original sequence.
- SC is the smallest such covering.

It is easy to prove the theorem that the dimensions of SC and LIS are the same, and it is possible to construct LIS from SC.

Next, let's suppose we have 2 strings of S1 and S2. It can be shown that if for each symbol in S1 we take the list of its appearances in S2 in the reverse order, and concatenate all such lists keeping order, then any IS in the resulting list will be equivalent to some common subsequence S1 and S2 of the same length. And, consequently, LIS will be equivalent to LCS.

The idea of the algorithm for constructing DS:
- Going along the original sequence, for the current member x in the list of its DS we find the leftmost, such that its last term is not less than x.
- If there is one, then add x to the end.
- If not, add a new DS consisting of x to the DS list.

It can be shown that the DS list constructed this way will be SC.

