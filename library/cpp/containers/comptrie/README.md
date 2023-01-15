Compact trie
=============

The comptrie library is a fast and very tightly packed
implementation of a prefix tree (Sedgewick's T-trie, that is a ternary tree,
see https://www.cs.princeton.edu/~rs/strings/paper.pdf,
https://www.cs.upc.edu/~ps/downloads/tst/tst.html). It contains tools for creating, optimizing, and serializing trees, accessing by key, and performing 
various searches. Because it is template-based and performance-oriented, a significant
part of the library consists of inline functions, so if you don't need all the
features of the library, consider including a more specific header file instead of the top-level
comptrie.h file.

Description of the data structure
---------------------------------

A prefix tree is an implementation of the map data structure
for cases when keys are sequences of characters. The nodes on this tree
contain characters and values. The key that corresponds to a
value is a sequence of all the characters on nodes along the path from the root to the
node with this value. It follows that a tree node can have as many children as
there are different characters, which is quite a lot. For compact storage and
quick access, the children are ordered as a binary search tree, which should be
balanced if possible (characters have a defined order that must be
preserved). Thus, our tree is ternary: each node has a reference to a subtree
of all children and two references to subtrees of siblings. One of these subtrees contains the
younger siblings, and the other contains the elder siblings.

The library implements tree optimization by merging identical subtrees, which means
the tree becomes a DAG (Directed Acyclic Graph –
an oriented graph without oriented cycles).

The main class TCompactTrie is defined in comptrie_trie.h and is templatized:
- The first parameter of the template is the character type. It should be an
integer type, which means that arithmetical operations must be defined for it.
- The second parameter of the template is the value type.
- The third parameter is the packer class, which packs values in order to quickly and compactly
serialize the value type to a continuous memory buffer, deserialize it
back, and quickly determine its size using the pointer to the beginning of this
memory buffer. Good packers have already been written for most types, and they are available in
library/cpp/packers. For more information, please refer to the documentation for these packers.

The set.h file defines a modification for cases when keys must be stored
without values.

When a tree is built from scratch, the value corresponding to an empty key is
assigned to a single-character key '\0'. So in a tree with the 'char' character type,
the empty key and the '\0' key are bound together. For a subtree received from
a call to FindTails, this restriction no longer exists.

Creating trees
--------------

Building a tree from a list of key-value pairs is performed by the
TCompactTrieBuilder class described in the comptrie_builder.h file.

This class allows you to add words to a tree one at a time, merge a complete
subtree, and also use an unfinished tree as a map.

An important optimization is the prefix-grouped mode when you need to add keys
in a certain order (for details, see the comments in the header file). The resulting tree is compactly packed while keys are being added, and the memory consumption is approximately the same as for
the completed tree. For the default mode, compact stacking is turned on at the
very end, and the data consumes quite a lot of memory up until that point.

Optimizing trees
----------------

After a tree is created, there are two optimizing operations that can be applied:
 - Minimization to a DAG by merging equal subtrees.
 - Fast memory layout.
The functions that implement these operations are declared in the comptrie_builder.h file. The first
optimization is implemented by the CompactTrieMinimize function, and the second is implemented by
CompactTrieMakeFastLayout. You can perform both at once by calling the
CompactTrieMinimizeAndMakeFastLayout function.

### Minimization ###

Minimization to a DAG requires quite a lot of time and a large amount of
memory to store an array of subtrees, but it can reduce the size of the tree several
times over (an example is a language model that has many low-frequency
phrases with repeated last words and frequency counts). However, if you know
in advance that there are no duplicate values in the tree, you don't need to waste time on it, since the minimization
won't have any effect on the tree.

### Fast memory layout ###

The second optimization function results in fewer cache misses, but it causes the
tree to grow in size. Our experience has shown a 5% gain
in speed for some tries. The algorithm consumes about three times more memory than
the amount required for the source tree. So if the machine has enough memory to
assemble a tree, it does not neccessarily mean that it has enough memory to run
the algorithm. To learn about the theory behind this algorithm, read the comments before the declaration of the CompactTrieMinimize function.

Serializing trees
-----------------

The tree resides in memory as a sequence of nodes. Links to other nodes are always
counted relative to the position of the current node. This allows you to save a
tree to disk as it is and then re-load it using mmap(). The TCompactTrie class has the
TBlob constructor for reading a tree from disk. The TCompactTrieBuilder class has
Save/SaveToFile methods for writing a built tree to a stream or a file.

Accessing trees
---------------

As a rule, all methods that accept a key as input have two variants:
- One takes the key in the format: pointer to the beginning of the key, length.
- The other takes a high-level type like TStringBuf.

You can get a value for a key in the tree – TCompactTrie::Find returns
false if there is no key, and TCompactTrie::Get throws an exception. You can use FindPrefix methods to find the longest key prefix in a tree and get the corresponding value for it.
You can also use a single FindPhrases request to get values for all the beginnings of
a phrase with a given word delimiter.

An important operation that distinguishes a tree from a simple map is implemented in the FindTails method, 
which allows you to obtain a subtree consisting of all possible extensions of the
given prefix.

Iterators for trees
-------------------

First of all, there is a typical map iterator over all key-value pairs called
TConstIterator. A tree has three methods that return it: Begin, End, and
UpperBound. The latter takes a key as input and returns an iterator to the
smallest key that is not smaller than the input key.

The rest of the iterators are not so widely used, and thus are located in
separate files.

TPrefixIterator is defined in the prefix_iterator.h file. It allows
iterations over all the prefixes of this key available in the tree.

TSearchIterator is defined in the search_iterator.h file. It allows you to enter
a key in a tree one character at a time and see where it ends up. The following character can
be selected depending on the current result. You can also copy the iterator and
proceed on two different paths. You can actually achieve the same result with
repeated use of the FindTails method, but the authors of this iterator claim
that they obtained a performance gain with it.

Appendix. Memory implementation details
---------------------------------------

*If you are not going to modify the library, then you do not need to read further.*

First, if the character type has a size larger than 1 byte, then all keys that use these characters are converted to byte strings in the big-endian way. This
means that character bytes are written in a string from the most significant
to the least significant from left to right. Thus it is reduced to the case when
the character in use is 'char'.

The tree resides in memory as a series of consecutive nodes. The nodes can have different
sizes, so the only way to identify the boundaries of nodes is by passing the entire
tree.

### Node structure ###

The structure of a node, as can be understood from thoughtfully reading the
LeapByte function in Comptrie_impl.h, is the following:
- The first byte is for service flags.
- The second byte is a character (unless it is the ε-link type of node
  described below, which has from 1 to 7 bytes of offset distance from the
  beginning of this node to the content node, and nothing else).

Thus, the size of any node is at least 2 bytes. All other elements of a node
are optional. Next there is from 0 to 7 bytes of the packed offset from the beginning
of this node to the beginning of the root node of a subtree with the younger
siblings. It is followed by 0 to 7 bytes of the packed offset from the beginning of this
node to the beginning of the root node of a subtree with the elder siblings.
Next comes the packed value in this node. Its size is not limited, but you may
recall that the packer allows you to quickly determine this size using a pointer
to the beginning of the packed value. Then, if the service flags indicate
that the tree has children, there is a root node of the subtree of children.

The packed offset is restricted to 7 bytes, and this gives us a limit on the largest 
possible size of a tree. You need to study the packer code to understand
the exact limit.

All packed offsets are nonnegative, meaning that roots of subtrees with
siblings and the node pointed to by the ε-link must be located
strictly to the right of the current node in memory. This does not allow placement of
finite state machines with oriented cycles in the comptrie. But it does allow you to
effectively stack the comptrie from right to left.

### Service flags ###

The byte of service flags contains (as shown by the constants at the beginning of
the comptrie_impl.h file):
- 1 bit of MT_NEXT, indicating whether this node has children.
- 1 bit of MT_FINAL, indicating if there is a value in this node.
- 3 bits of MT_SIZEMASK, indicating the size of the packed offset to a subtree
  with elder siblings.
- 3 bits of MT_SIZEMASK << MT_LEFTSHIFT, indicating the size of the packed
  offset to a subtree with younger siblings.
If one of these subtrees is not present, then the size of the corresponding
packed offset is 0, and vice versa.

### ε-links ###

These nodes only occur if we optimized a tree into a DAG and got two nodes with
merged subtrees of children. Since the offset to the subtree of children can't be
specified and the root of this subtree should lie just after the value, we have
to add a node of the ε-link type, which contains the offset to the root subtree of
children and nothing more. This applies to all nodes that have equal subtrees of children,
except the rightmost node. The size of this offset is set in 3 bits of MT_SIZEMASK
flags for a node.

As the implementation of the IsEpsilonLink function in
comptrie_impl.h demonstrates, the ε-link differs from other nodes in that it does not have the MT_NEXT flag or the MT_FINAL
 flag, so it can always be
identified by the flags. Of course, the best programming practice is to call the
function itself instead of examining the flags.

Note that the ε-link flags do not use the MT_SIZEMASK <<
MT_LEFTSHIFT` bits, which allows us to start using ε-links for some other purpose.

Pattern Searcher
================

This is an implementation of Aho-Corasick algorithm on compact trie structure.
In order to create a pattern searcher one must fill a TCompactPatternSearcherBuilder
with patterns and call SaveAsPatternSearcher or SaveToFileAsPatternSearcher.
Then TCompactPatternSearcher must be created from the builder output.

### Implementation details ###

Aho-Corasick algorithm stores a suffix link in each node.
A suffix link of a node is the offset (relative to this node) of the largest suffix
of a string this node represents which is present in a trie.
Current implementation also stores a shortcut link to the largest suffix
for which the corresponding node in a trie is a final node.
These two links are stored as NCompactTrie::TSuffixLink structure of two 64-bit
integers.
In a trie layout these links are stored for each node right after the two bytes
containing service flags and a symbol.
