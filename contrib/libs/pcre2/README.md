# PCRE2 - Perl-Compatible Regular Expressions

The PCRE2 library is a set of C functions that implement regular expression
pattern matching using the same syntax and semantics as Perl 5. PCRE2 has its
own native API, as well as a set of wrapper functions that correspond to the
POSIX regular expression API. The PCRE2 library is free, even for building 
proprietary software. It comes in three forms, for processing 8-bit, 16-bit,
or 32-bit code units, in either literal or UTF encoding.

PCRE2 was first released in 2015 to replace the API in the original PCRE 
library, which is now obsolete and no longer maintained. As well as a more
flexible API, the code of PCRE2 has been much improved since the fork.
 
## Download

As well as downloading from the 
[GitHub site](https://github.com/PCRE2Project/pcre2), you can download PCRE2 
or the older, unmaintained PCRE1 library from an 
[*unofficial* mirror](https://sourceforge.net/projects/pcre/files/) at SourceForge.

You can check out the PCRE2 source code via Git or Subversion:

    git clone https://github.com/PCRE2Project/pcre2.git
    svn co    https://github.com/PCRE2Project/pcre2.git

## Contributed Ports

If you just need the command-line PCRE2 tools on Windows, precompiled binary
versions are available at this 
[Rexegg page](http://www.rexegg.com/pcregrep-pcretest.html).

A PCRE2 port for z/OS, a mainframe operating system which uses EBCDIC as its
default character encoding, can be found at 
[http://www.cbttape.org](http://www.cbttape.org/) (File 939).

## Documentation

You can read the PCRE2 documentation 
[here](https://PCRE2Project.github.io/pcre2/doc/html/index.html).

Comparisons to Perl's regular expression semantics can be found in the
community authored Wikipedia entry for PCRE.

There is a curated summary of changes for each PCRE release, copies of
documentation from older releases, and other useful information from the third
party authored 
[RexEgg PCRE Documentation and Change Log page](http://www.rexegg.com/pcre-documentation.html).

## Contact

To report a problem with the PCRE2 library, or to make a feature request, please
use the PCRE2 GitHub issues tracker. There is a mailing list for discussion of
 PCRE2 issues and development at pcre2-dev@googlegroups.com, which is where any
announcements will be made. You can browse the 
[list archives](https://groups.google.com/g/pcre2-dev).

