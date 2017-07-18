`library` - is a folder with a flat[1] list of libraries. Each library should
confirm to the following set of criteria:

1.  It's an application[2] library.

2.  The library must be in use in at least two projects.

3.  Code of the library must comply to style guide, and be portable

    (use our cross-platform framework).
4.  The library itself is not a part of the cross-platform framework

    (it shouldn't contain private utils and commons in it).

5.  The library must depend only on a limited list of external components

    (currently it is `util`, `contrib`, `library`).

6.  Any code could depend on the library except `contrib`, `util`.

7.  The library must be accompanied with: `README.md` file with a brief

    description of the library, unit-tests and benchmarks.



[1] Libraries of similar type could be grouped into subfolders. For example:

* container       - containers (vectors, etc.)
* hashes          - different implementations of hash
* xml             - XML utils (parsers, SAX-parsers, etc.)

[2] To get an understanding what is an application and platform library see the description below:

Example what is in the library:

* processing of keyinv-files of inverted index
* neh, messagebus - message exchange libraries
* library of regular expressions pire

What will never appear here:

* a new super-mega string splitter
* a new super-mega iterator over directories

Reasoning: To stimulate use of common components and common framework.
The idea is simple - if you want your code to be widely used, then do not create
a new common, since it will not be possible to put the library into an
off-project repository. And avoid false dependencies between projects.

We understand that there will always be a fine line between the platform and
applied libraries, and disputable points will be decided on the basis of
common sense.
