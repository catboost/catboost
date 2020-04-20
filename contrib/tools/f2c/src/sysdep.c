/****************************************************************
Copyright 1990 - 1994, 2000 by AT&T, Lucent Technologies and Bellcore.

Permission to use, copy, modify, and distribute this software
and its documentation for any purpose and without fee is hereby
granted, provided that the above copyright notice appear in all
copies and that both that the copyright notice and this
permission notice and warranty disclaimer appear in supporting
documentation, and that the names of AT&T, Bell Laboratories,
Lucent or Bellcore or any of their entities not be used in
advertising or publicity pertaining to distribution of the
software without specific, written prior permission.

AT&T, Lucent and Bellcore disclaim all warranties with regard to
this software, including all implied warranties of
merchantability and fitness.  In no event shall AT&T, Lucent or
Bellcore be liable for any special, indirect or consequential
damages or any damages whatsoever resulting from loss of use,
data or profits, whether in an action of contract, negligence or
other tortious action, arising out of or in connection with the
use or performance of this software.
****************************************************************/
#include "defs.h"
#include "usignal.h"
#include <stdlib.h>

char binread[] = "rb", textread[] = "r";
char binwrite[] = "wb", textwrite[] = "w";
char *c_functions	= "c_functions";
char *coutput		= "c_output";
char *initfname		= "raw_data";
char *initbname		= "raw_data.b";
char *blkdfname		= "block_data";
char *p1_file		= "p1_file";
char *p1_bakfile	= "p1_file.BAK";
char *sortfname		= "init_file";
char *proto_fname	= "proto_file";

char link_msg[]	= "on Microsoft Windows system, link with libf2c.lib;\n\
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm\n\
	or, if you install libf2c.a in a standard place, with -lf2c -lm\n\
	-- in that order, at the end of the command line, as in\n\
		cc *.o -lf2c -lm\n\
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,\n\n\
		http://www.netlib.org/f2c/libf2c.zip";

char *outbuf = "", *outbtail;

#undef WANT_spawnvp
#ifdef MSDOS
#ifndef NO_spawnvp
#define WANT_spawnvp
#endif
#endif

#ifdef _WIN32
#include <windows.h>	/* for GetVolumeInformation */
#undef WANT_spawnvp
#define WANT_spawnvp
#undef  MSDOS
#define MSDOS
#endif

#ifdef WANT_spawnvp
#include <process.h>
#ifndef _P_WAIT
#define _P_WAIT P_WAIT	/* Symantec C/C++ */
#endif
static char **spargv, **pfname;
#endif

char *tmpdir = "";

#ifdef __cplusplus
#define Cextern extern "C"
extern "C" {
 static void flovflo(int), killed(int);
 static int compare(const void *a, const void *b);
}
#else
#define Cextern extern
#endif

Cextern int unlink Argdcl((const char *));
Cextern int fork Argdcl((void)), getpid Argdcl((void)), wait Argdcl((int*));

 void
#ifdef KR_headers
Un_link_all(cdelete)
	int cdelete;
#else
Un_link_all(int cdelete)
#endif
{
	if (!debugflag) {
		unlink(c_functions);
		unlink(initfname);
                unlink(initbname);
		unlink(p1_file);
                unlink(p1_bakfile);
		unlink(sortfname);
		unlink(blkdfname);
		if (cdelete && coutput)
			unlink(coutput);
		}
	}

#ifndef NO_TEMPDIR
 static void
rmtdir(Void)
{
	char *s;
	if (*(s = tmpdir)) {
		tmpdir = "";
		rmdir(s);
		}
	}
#endif /*NO_TEMPDIR*/

#ifndef MSDOS
#include "sysdep.hd"
#ifndef NO_MKDTEMP
#include <unistd.h> /* for mkdtemp */
#endif
#endif

 static void
alloc_names(Void)
{
	int k = strlen(tmpdir) + 24;
	c_functions = (char *)ckalloc(7*k);
	initfname = c_functions + k;
	initbname = initfname + k;
	blkdfname = initbname + k;
	p1_file = blkdfname + k;
	p1_bakfile = p1_file + k;
	sortfname = p1_bakfile + k;
	}

 void
set_tmp_names(Void)
{
#ifdef MSDOS
	char buf[64], *s, *t;
#ifdef _WIN32
	DWORD flags, maxlen, volser;
	char volname[512], f2c[24], fsname[512], *name1;
	int i;

	if (debugflag == 1)
		return;
	i = sprintf(f2c, "%x", _getpid());
	if (!GetVolumeInformation(NULL, volname, sizeof(volname), &volser, &maxlen,
			&flags, fsname, sizeof(fsname))
	 || maxlen < i+8) /* FAT16 */
		strcpy(f2c, "f2c_");
#else
	static char f2c[] = "f2c_";
	if (debugflag == 1)
		return;
#endif

	if (!*tmpdir || *tmpdir == '.' && !tmpdir[1])
		t = "";
	else {
		/* substitute \ for / to avoid confusion with a
		 * switch indicator in the system("sort ...")
		 * call in formatdata.c
		 */
		for(s = tmpdir, t = buf; *s; s++, t++)
			if ((*t = *s) == '/')
				*t = '\\';
		if (t[-1] != '\\')
			*t++ = '\\';
		*t = 0;
		t = buf;
		}
	alloc_names();
	sprintf(c_functions, "%s%sfunc", t, f2c);
	sprintf(initfname, "%s%srd", t, f2c);
	sprintf(blkdfname, "%s%sblkd", t, f2c);
	sprintf(p1_file, "%s%sp1f", t, f2c);
	sprintf(p1_bakfile, "%s%sp1fb", t, f2c);
	sprintf(sortfname, "%s%ssort", t, f2c);
#else /*!MSDOS*/
	long pid;

#define L_TDNAME 20
#ifdef NO_MKDTEMP
#ifdef NO_MKSTEMP
#undef  L_TDNAME
#define L_TDNAME L_tmpnam
#endif
#endif
	static char tdbuf[L_TDNAME];

	if (debugflag == 1)
		return;
	pid = getpid();
	if (!*tmpdir) {
#ifdef NO_TEMPDIR
		tmpdir = "/tmp";
#else
#ifdef NO_MKDTEMP
#ifdef NO_MKSTEMP
		if (!(tmpdir = tmpnam(tdbuf))) {
			fprintf(stderr, "tmpnam failed (for -T)\n");
			exit(1);
			}
#else
		int f;
		strcpy(tdbuf, "/tmp/f2ctd_XXXXXX");
		f = mkstemp(tdbuf);
		if (f >= 0) {
			close(f);
			remove(tmpdir = tdbuf);
			}
		else {
			fprintf(stderr, "mkstemp failed (for -T)\n");
			exit(1);
			}
#endif /*NO_MKSTEMP*/
		if (mkdir(tdbuf,0700)) {
			fprintf(stderr, "mkdir failed (for -T)\n");
			exit(1);
			}
#else /*!NO_MKDTEMP*/
		strcpy(tdbuf, "/tmp/f2ctd_XXXXXX");
		if (!(tmpdir = mkdtemp(tdbuf))) {
			fprintf(stderr, "mkdtemp failed (for -T)\n");
			exit(1);
			}
#endif /*NO_MKDTEMP*/
		if (!debugflag)
			atexit(rmtdir);
#endif /*NO_TEMPDIR*/
		}
	alloc_names();
        /* What follows is safe if tmpdir is really
        a private diectory created by us -- otherwise
        the file could be a sym link somewhere else....*/
	sprintf(c_functions, "%s/f2c%ld_func", tmpdir, pid);
	sprintf(initfname, "%s/f2c%ld_rd", tmpdir, pid);
	sprintf(blkdfname, "%s/f2c%ld_blkd", tmpdir, pid);
	sprintf(p1_file, "%s/f2c%ld_p1f", tmpdir, pid);
	sprintf(p1_bakfile, "%s/f2c%ld_p1fb", tmpdir, pid);
	sprintf(sortfname, "%s/f2c%ld_sort", tmpdir, pid);
#endif /*MSDOS*/
	sprintf(initbname, "%s.b", initfname);
	if (debugflag)
		fprintf(diagfile, "%s %s %s %s %s %s\n", c_functions,
			initfname, blkdfname, p1_file, p1_bakfile, sortfname);
	}

 char *
#ifdef KR_headers
c_name(s, ft)
	char *s;
	int ft;
#else
c_name(char *s, int ft)
#endif
{
	char *b, *s0;
	int c;

	b = s0 = s;
	while(c = *s++)
		if (c == '/')
			b = s;
	if (--s < s0 + 3 || s[-2] != '.'
			 || ((c = *--s) != 'f' && c != 'F')) {
		infname = s0;
		Fatal("file name must end in .f or .F");
		}
	strcpy(outbtail, b);
	outbtail[s-b] = ft;
	b = copys(outbuf);
	return b;
	}

 static void
#ifdef KR_headers
killed(sig)
	int sig;
#else
killed(int sig)
#endif
{
	sig = sig;	/* shut up warning */
	signal(SIGINT, SIG_IGN);
#ifdef SIGQUIT
	signal(SIGQUIT, SIG_IGN);
#endif
#ifdef SIGHUP
	signal(SIGHUP, SIG_IGN);
#endif
	signal(SIGTERM, SIG_IGN);
	Un_link_all(1);
	exit(126);
	}

 static void
#ifdef KR_headers
sig1catch(sig)
	int sig;
#else
sig1catch(int sig)
#endif
{
	sig = sig;	/* shut up warning */
	if (signal(sig, SIG_IGN) != SIG_IGN)
		signal(sig, killed);
	}

 static void
#ifdef KR_headers
flovflo(sig)
	int sig;
#else
flovflo(int sig)
#endif
{
	sig = sig;	/* shut up warning */
	Fatal("floating exception during constant evaluation; cannot recover");
	/* vax returns a reserved operand that generates
	   an illegal operand fault on next instruction,
	   which if ignored causes an infinite loop.
	*/
	signal(SIGFPE, flovflo);
}

 void
#ifdef KR_headers
sigcatch(sig)
	int sig;
#else
sigcatch(int sig)
#endif
{
	sig = sig;	/* shut up warning */
	sig1catch(SIGINT);
#ifdef SIGQUIT
	sig1catch(SIGQUIT);
#endif
#ifdef SIGHUP
	sig1catch(SIGHUP);
#endif
	sig1catch(SIGTERM);
	signal(SIGFPE, flovflo);  /* catch overflows */
	}

/* argkludge permits wild-card expansion and caching of the original or expanded */
/* argv to kludge around the lack of fork() and exec() when necessary. */

 void
#ifdef KR_headers
argkludge(pargc, pargv) int *pargc; char ***pargv;
#else
argkludge(int *pargc, char ***pargv)
#endif
{
#ifdef WANT_spawnvp
	size_t L, L1;
	int argc, i, nf;
	char **a, **argv, *s, *t, *t0;

	/* Assume wild-card expansion has been done by Microsoft's setargv.obj */

	/* Count Fortran input files. */

	L = argc = *pargc;
	argv = *pargv;
	for(i = nf = 0; i < argc; i++) {
		L += L1 = strlen(s = argv[i]);
		if (L1 > 2 && s[L1-2] == '.')
			switch(s[L1-1]) {
			  case 'f':
			  case 'F':
				nf++;
			  }
		}
	if (nf <= 1)
		return;

	/* Cache inputs */

	i = argc - nf + 2;
	a = spargv = (char**)Alloc(i*sizeof(char*) + L);
	t = (char*)(a + i);
	for(i = 0; i < argc; i++) {
		*a++ = t0 = t;
		for(s = argv[i]; *t++ = *s; s++);
		if (t-t0 > 3 && s[-2] == '.')
			switch(s[-1]) {
			  case 'f':
			  case 'F':
				--a;
				t = t0;
			  }
		}
	pfname = a++;
	*a = 0;
#endif
	}

 int
#ifdef KR_headers
dofork(fname) char *fname;
#else
dofork(char *fname)
#endif
{
	extern int retcode;
#ifdef MSDOS
#ifdef WANT_spawnvp
	*pfname = fname;
	retcode |= _spawnvp(_P_WAIT, spargv[0], (char const*const*)spargv);
#else /*_WIN32*/
	Fatal("Only one Fortran input file allowed under MS-DOS");
#endif /*_WIN32*/
#else
	int pid, status, w;

	if (!(pid = fork()))
		return 1;
	if (pid == -1)
		Fatal("bad fork");
	while((w = wait(&status)) != pid)
		if (w == -1)
			Fatal("bad wait code");
	retcode |= status >> 8;
#endif
	return 0;
	}

/* Initialization of tables that change with the character set... */

char escapes[Table_size];

#ifdef non_ASCII
char *str_fmt[Table_size];
static char *str0fmt[127] = { /*}*/
#else
char *str_fmt[Table_size] = {
#endif
 "\\000", "\\001", "\\002", "\\003", "\\004", "\\005", "\\006", "\\007",
   "\\b",   "\\t",   "\\n", "\\013",   "\\f",   "\\r", "\\016", "\\017",
 "\\020", "\\021", "\\022", "\\023", "\\024", "\\025", "\\026", "\\027",
 "\\030", "\\031", "\\032", "\\033", "\\034", "\\035", "\\036", "\\037",
     " ",     "!",  "\\\"",     "#",     "$",     "%%",    "&",     "'",
     "(",     ")",     "*",     "+",     ",",     "-",     ".",     "/",
     "0",     "1",     "2",     "3",     "4",     "5",     "6",     "7",
     "8",     "9",     ":",     ";",     "<",     "=",     ">",     "?",
     "@",     "A",     "B",     "C",     "D",     "E",     "F",     "G",
     "H",     "I",     "J",     "K",     "L",     "M",     "N",     "O",
     "P",     "Q",     "R",     "S",     "T",     "U",     "V",     "W",
     "X",     "Y",     "Z",     "[",  "\\\\",     "]",     "^",     "_",
     "`",     "a",     "b",     "c",     "d",     "e",     "f",     "g",
     "h",     "i",     "j",     "k",     "l",     "m",     "n",     "o",
     "p",     "q",     "r",     "s",     "t",     "u",     "v",     "w",
     "x",     "y",     "z",     "{",     "|",     "}",     "~"
     };

#ifdef non_ASCII
char *chr_fmt[Table_size];
static char *chr0fmt[127] = {	/*}*/
#else
char *chr_fmt[Table_size] = {
#endif
   "\\0",   "\\1",   "\\2",   "\\3",   "\\4",   "\\5",   "\\6",   "\\7",
   "\\b",   "\\t",   "\\n",  "\\13",   "\\f",   "\\r",  "\\16",  "\\17",
  "\\20",  "\\21",  "\\22",  "\\23",  "\\24",  "\\25",  "\\26",  "\\27",
  "\\30",  "\\31",  "\\32",  "\\33",  "\\34",  "\\35",  "\\36",  "\\37",
     " ",     "!",    "\"",     "#",     "$",     "%%",    "&",   "\\'",
     "(",     ")",     "*",     "+",     ",",     "-",     ".",     "/",
     "0",     "1",     "2",     "3",     "4",     "5",     "6",     "7",
     "8",     "9",     ":",     ";",     "<",     "=",     ">",     "?",
     "@",     "A",     "B",     "C",     "D",     "E",     "F",     "G",
     "H",     "I",     "J",     "K",     "L",     "M",     "N",     "O",
     "P",     "Q",     "R",     "S",     "T",     "U",     "V",     "W",
     "X",     "Y",     "Z",     "[",  "\\\\",     "]",     "^",     "_",
     "`",     "a",     "b",     "c",     "d",     "e",     "f",     "g",
     "h",     "i",     "j",     "k",     "l",     "m",     "n",     "o",
     "p",     "q",     "r",     "s",     "t",     "u",     "v",     "w",
     "x",     "y",     "z",     "{",     "|",     "}",     "~"
     };

 void
fmt_init(Void)
{
	static char *str1fmt[6] =
		{ "\\b", "\\t", "\\n", "\\f", "\\r", "\\013" };
	register int i, j;
	register char *s;

	/* str_fmt */

#ifdef non_ASCII
	i = 0;
#else
	i = 127;
#endif
	s = Alloc(5*(Table_size - i));
	for(; i < Table_size; i++) {
		sprintf(str_fmt[i] = s, "\\%03o", i);
		s += 5;
		}
#ifdef non_ASCII
	for(i = 32; i < 127; i++) {
		s = str0fmt[i];
		str_fmt[*(unsigned char *)s] = s;
		}
	str_fmt['"'] = "\\\"";
#else
	if (Ansi == 1)
		str_fmt[7] = chr_fmt[7] = "\\a";
#endif

	/* chr_fmt */

#ifdef non_ASCII
	for(i = 0; i < 32; i++)
		chr_fmt[i] = chr0fmt[i];
#else
	i = 127;
#endif
	for(; i < Table_size; i++)
		chr_fmt[i] = "\\%o";
#ifdef non_ASCII
	for(i = 32; i < 127; i++) {
		s = chr0fmt[i];
		j = *(unsigned char *)s;
		if (j == '\\')
			j = *(unsigned char *)(s+1);
		chr_fmt[j] = s;
		}
#endif

	/* escapes (used in lex.c) */

	for(i = 0; i < Table_size; i++)
		escapes[i] = i;
	for(s = "btnfr0", i = 0; i < 6; i++)
		escapes[*(unsigned char *)s++] = "\b\t\n\f\r"[i];
	/* finish str_fmt and chr_fmt */

	if (Ansi)
		str1fmt[5] = "\\v";
	if ('\v' == 'v') { /* ancient C compiler */
		str1fmt[5] = "v";
#ifndef non_ASCII
		escapes['v'] = 11;
#endif
		}
	else
		escapes['v'] = '\v';
	for(s = "\b\t\n\f\r\v", i = 0; j = *(unsigned char *)s++;)
		str_fmt[j] = chr_fmt[j] = str1fmt[i++];
	/* '\v' = 11 for both EBCDIC and ASCII... */
	chr_fmt[11] = (char*)(Ansi ? "\\v" : "\\13");
	}

 void
outbuf_adjust(Void)
{
	int n, n1;
	char *s;

	n = n1 = strlen(outbuf);
	if (*outbuf && outbuf[n-1] != '/')
		n1++;
	s = Alloc(n+64);
	outbtail = s + n1;
	strcpy(s, outbuf);
	if (n != n1)
		strcpy(s+n, "/");
	outbuf = s;
	}


/* Unless SYSTEM_SORT is defined, the following gives a simple
 * in-core version of dsort().  On Fortran source with huge DATA
 * statements, the in-core version may exhaust the available memory,
 * in which case you might either recompile this source file with
 * SYSTEM_SORT defined (if that's reasonable on your system), or
 * replace the dsort below with a more elaborate version that
 * does a merging sort with the help of auxiliary files.
 */

#ifdef SYSTEM_SORT

 int
#ifdef KR_headers
dsort(from, to)
	char *from;
	char *to;
#else
dsort(char *from, char *to)
#endif
{
	char buf[200];
	sprintf(buf, "sort <%s >%s", from, to);
	return system(buf) >> 8;
	}
#else

 static int
#ifdef KR_headers
 compare(a,b)
  char *a, *b;
#else
 compare(const void *a, const void *b)
#endif
{ return strcmp(*(char **)a, *(char **)b); }

 int
#ifdef KR_headers
dsort(from, to)
	char *from;
	char *to;
#else
dsort(char *from, char *to)
#endif
{
	struct Memb {
		struct Memb *next;
		int n;
		char buf[32000];
		};
	typedef struct Memb memb;
	memb *mb, *mb1;
	register char *x, *x0, *xe;
	register int c, n;
	FILE *f;
	char **z, **z0;
	int nn = 0;

	f = opf(from, textread);
	mb = (memb *)Alloc(sizeof(memb));
	mb->next = 0;
	x0 = x = mb->buf;
	xe = x + sizeof(mb->buf);
	n = 0;
	for(;;) {
		c = getc(f);
		if (x >= xe && (c != EOF || x != x0)) {
			if (!n)
				return 126;
			nn += n;
			mb->n = n;
			mb1 = (memb *)Alloc(sizeof(memb));
			mb1->next = mb;
			mb = mb1;
			memcpy(mb->buf, x0, n = x-x0);
			x0 = mb->buf;
			x = x0 + n;
			xe = x0 + sizeof(mb->buf);
			n = 0;
			}
		if (c == EOF)
			break;
		if (c == '\n') {
			++n;
			*x++ = 0;
			x0 = x;
			}
		else
			*x++ = c;
		}
	clf(&f, from, 1);
	f = opf(to, textwrite);
	if (x > x0) { /* shouldn't happen */
		*x = 0;
		++n;
		}
	mb->n = n;
	nn += n;
	if (!nn) /* shouldn't happen */
		goto done;
	z = z0 = (char **)Alloc(nn*sizeof(char *));
	for(mb1 = mb; mb1; mb1 = mb1->next) {
		x = mb1->buf;
		n = mb1->n;
		for(;;) {
			*z++ = x;
			if (--n <= 0)
				break;
			while(*x++);
			}
		}
	qsort((char *)z0, nn, sizeof(char *), compare);
	for(n = nn, z = z0; n > 0; n--)
		fprintf(f, "%s\n", *z++);
	free((char *)z0);
 done:
	clf(&f, to, 1);
	do {
		mb1 = mb->next;
		free((char *)mb);
		}
		while(mb = mb1);
	return 0;
	}
#endif
