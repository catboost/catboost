To compile f2c on Linux or Unix systems, copy makefile.u to makefile,
edit makefile if necessary (see the comments in it and below) and
type "make" (or maybe "nmake", depending on your system).

To compile f2c.exe on MS Windows systems with Microsoft Visual C++,

	copy makefile.vc makefile
	nmake

With other PC compilers, you may need to compile xsum.c with -DMSDOS
(i.e., with MSDOS #defined).

If your compiler does not understand ANSI/ISO C syntax (i.e., if
you have a K&R C compiler), compile with -DKR_headers .

On non-Unix systems where files have separate binary and text modes,
you may need to "make xsumr.out" rather than "make xsum.out".

If (in accordance with what follows) you need to any of the source
files (excluding the makefile), first issue a "make xsum.out" (or, if
appropriate, "make xsumr.out") to check the validity of the f2c source,
then make your changes, then type "make f2c".

The file usignal.h is for the benefit of strictly ANSI include files
on a UNIX system -- the ANSI signal.h does not define SIGHUP or SIGQUIT.
You may need to modify usignal.h if you are not running f2c on a UNIX
system.

Should you get the message "xsum0.out xsum1.out differ", see what lines
are different (`diff xsum0.out xsum1.out`) and ask netlib
(e.g., netlib@netlib.org) to send you the files in question,
plus the current xsum0.out (which may have changed) "from f2c/src".
For example, if exec.c and expr.c have incorrect check sums, you would
send netlib the message
	send exec.c expr.c xsum0.out from f2c/src
You can also ftp these files from netlib.bell-labs.com; for more
details, ask netlib@netlib.org to "send readme from f2c".

On some systems, the malloc and free in malloc.c let f2c run faster
than do the standard malloc and free.  Other systems may not tolerate
redefinition of malloc and free (though changes of 8 Nov. 1994 may
render this less of a problem than hitherto).  If your system permits
use of a user-supplied malloc, you may wish to change the MALLOC =
line in the makefile to "MALLOC = malloc.o", or to type
	make MALLOC=malloc.o
instead of
	make
Still other systems have a -lmalloc that provides performance
competitive with that from malloc.c; you may wish to compare the two
on your system.  If your system does not permit user-supplied malloc
routines, then f2c may fault with "MALLOC=malloc.o", or may display
other untoward behavior.

On some BSD systems, you may need to create a file named "string.h"
whose single line is
#include <strings.h>
you may need to add " -Dstrchr=index" to the "CFLAGS =" assignment
in the makefile, and you may need to add " memset.o" to the "OBJECTS ="
assignment in the makefile -- see the comments in memset.c .

For non-UNIX systems, you may need to change some things in sysdep.c,
such as the choice of intermediate file names.

On some systems, you may need to modify parts of sysdep.h (which is
included by defs.h).  In particular, for Sun 4.1 systems and perhaps
some others, you need to comment out the typedef of size_t.  For some
systems (e.g., IRIX 4.0.1 and AIX) it is better to add
#define ANSI_Libraries
to the beginning of sysdep.h (or to supply -DANSI_Libraries in the
makefile).

Alas, some systems #define __STDC__ but do not provide a true standard
(ANSI or ISO) C environment, e.g. do not provide stdlib.h .  If yours
is such a system, then (a) you should complain loudly to your vendor
about __STDC__ being erroneously defined, and (b) you should insert
#undef __STDC__
at the beginning of sysdep.h .  You may need to make other adjustments.

For some non-ANSI versions of stdio, you must change the values given
to binread and binwrite in sysdep.c from "rb" and "wb" to "r" and "w".
You may need to make this change if you run f2c and get an error
message of the form
	Compiler error ... cannot open intermediate file ...

In the days of yore, two libraries, libF77 and libI77, were used with
f77 (the Fortran compiler on which f2c is based).  Separate source for
these libraries is still available from netlib, but it is more
convenient to combine them into a single library, libf2c.  Source for
this combined library is also available from netlib in f2c/libf2c.zip,
e.g.,
	http://netlib.bell-labs.com/netlib/f2c/libf2c.zip
or
	http://www.netlib.org/f2c/libf2c.zip

(and similarly for other netlib mirrors).  After unzipping libf2c.zip,
copy the relevant makefile.* to makefile, edit makefile if necessary
(see the comments in it and in libf2c/README) and invoke "make" or
"nmake".  The resulting library is called *f2c.lib on MS Windows
systems and libf2c.a or libf2c.so on Linux and Unix systems;
makefile.u just shows how to make libf2c.a.  Details on creating the
shared-library variant, libf2c.so, are system-dependent; some that
have worked under Linux appear below.  For some other systems, you can
glean the details from the system-dependent makefile variants in
directory http://www.netlib.org/ampl/solvers/funclink or
http://netlib.bell-labs.com/netlib/ampl/solvers/funclink, etc.

In general, under Linux it is necessary to compile libf2c (or libI77)
with -DNON_UNIX_STDIO .  Under at least one variant of Linux, you can
make and install a shared-library version of libf2c by compiling
libI77 with -DNON_UNIX_STDIO, creating libf2c.a as above, and then
executing

	mkdir t
	ln lib?77/*.o t
	cd t; cc -shared -o ../libf2c.so -Wl,-soname,libf2c.so.1 *.o
	cd ..
	rm -r t
	rm /usr/lib/libf2c*
	mv libf2c.a libf2c.so /usr/lib
	cd /usr/lib
	ln libf2c.so libf2c.so.1
	ln libf2c.so libf2c.so.1.0.0

On some other systems, /usr/local/lib is the appropriate installation
directory.


Some older C compilers object to
	typedef void (*foo)();
or to
	typedef void zap;
	zap (*foo)();
If yours is such a compiler, change the definition of VOID in
f2c.h from void to int.

For convenience with systems that use control-Z to denote end-of-file,
f2c treats control-Z characters (ASCII 26, '\x1a') that appear at the
beginning of a line as an end-of-file indicator.  You can disable this
test by compiling lex.c with NO_EOF_CHAR_CHECK #defined, or can
change control-Z to some other character by #defining EOF_CHAR to
be the desired value.


If your machine has IEEE, VAX, or IBM-mainframe arithmetic, but your
printf is inaccurate (e.g., with Symantec C++ version 6.0,
printf("%.17g",12.) prints 12.000000000000001), you can make f2c print
correctly rounded numbers by compiling with -DUSE_DTOA and adding
dtoa.o g_fmt.o to the makefile's OBJECTS = line, so it becomes

	OBJECTS = $(OBJECTSd) malloc.o dtoa.o g_fmt.o

Also add the rule

	dtoa.o: dtoa.c
		$(CC) -c $(CFLAGS) -DMALLOC=ckalloc -DIEEE... dtoa.c

(without the initial tab) to the makefile, where IEEE... is one of
IEEE_MC68k, IEEE_8087, VAX, or IBM, depending on your machine's
arithmetic.  See the comments near the start of dtoa.c.

The relevant source files, dtoa.c and g_fmt.c, are available
separately from netlib's fp directory.  For example, you could
send the E-mail message

	send dtoa.c g_fmt.c from fp

to netlib@netlib.netlib.org (or use anonymous ftp from
ftp.netlib.org and look in directory /netlib/fp).

The makefile has a rule for creating tokdefs.h.  If you cannot use the
makefile, an alternative is to extract tokdefs.h from the beginning of
gram.c: it's the first 100 lines.

File mem.c has #ifdef CRAY lines that are appropriate for machines
with the conventional CRAY architecture, but not for "Cray" machines
based on DEC Alpha chips, such as the T3E; on such machines, you may
need to make a suitable adjustment, e.g., add #undef CRAY to sysdep.h.


Please send bug reports to dmg at acm.org (with " at " changed to "@").
The old index file (now called "readme" due to unfortunate changes in
netlib conventions:  "send readme from f2c") will report recent
changes in the recent-change log at its end; all changes will be shown
in the "changes" file ("send changes from f2c").  To keep current
source, you will need to request xsum0.out and version.c, in addition
to the changed source files.
