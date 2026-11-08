Version 0.3.3
=============

- Remove dead code, including unused feature `unstable-thread-sea`

Version 0.3.2
=============

- Add missing inlines to a few trivial methods (affects .top())

Version 0.3.1
=============

- `.join(a, b)` now allows both jobs a, b to panic and panics are propagated to
  the caller.
- Updated job completion flag to use release and acquire orderings.

Version 0.3.0
=============

- Cleanup of the public api for ThreadTree (removing unused methods), and use
  Box instead of Arc
- Add experimental convenience methods join3l, join3r, join4

Version 0.2.0
=============

First crates.io version.

New features
------------

- ThreadTree, the tree-structured thread pool by [@bluss]
