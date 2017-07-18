#include "f2c.h"
#include "fio.h"

/* Compile this with -DNO_TRUNCATE if unistd.h does not exist or */
/* if it does not define int truncate(const char *name, off_t). */

#ifdef MSDOS
#undef NO_TRUNCATE
#define NO_TRUNCATE
#endif

#ifndef NO_TRUNCATE
#include "unistd.h"
#endif

#ifdef KR_headers
extern char *strcpy();
extern FILE *tmpfile();
#else
#undef abs
#undef min
#undef max
#include "stdlib.h"
#include "string.h"
#ifdef __cplusplus
extern "C" {
#endif
#endif

extern char *f__r_mode[], *f__w_mode[];

#ifdef KR_headers
integer f_end(a) alist *a;
#else
integer f_end(alist *a)
#endif
{
	unit *b;
	FILE *tf;

	if(a->aunit>=MXUNIT || a->aunit<0) err(a->aerr,101,"endfile");
	b = &f__units[a->aunit];
	if(b->ufd==NULL) {
		char nbuf[10];
		sprintf(nbuf,"fort.%ld",(long)a->aunit);
		if (tf = FOPEN(nbuf, f__w_mode[0]))
			fclose(tf);
		return(0);
		}
	b->uend=1;
	return(b->useek ? t_runc(a) : 0);
}

#ifdef NO_TRUNCATE
 static int
#ifdef KR_headers
copy(from, len, to) FILE *from, *to; register long len;
#else
copy(FILE *from, register long len, FILE *to)
#endif
{
	int len1;
	char buf[BUFSIZ];

	while(fread(buf, len1 = len > BUFSIZ ? BUFSIZ : (int)len, 1, from)) {
		if (!fwrite(buf, len1, 1, to))
			return 1;
		if ((len -= len1) <= 0)
			break;
		}
	return 0;
	}
#endif /* NO_TRUNCATE */

 int
#ifdef KR_headers
t_runc(a) alist *a;
#else
t_runc(alist *a)
#endif
{
	OFF_T loc, len;
	unit *b;
	int rc;
	FILE *bf;
#ifdef NO_TRUNCATE
	FILE *tf;
#endif

	b = &f__units[a->aunit];
	if(b->url)
		return(0);	/*don't truncate direct files*/
	loc=FTELL(bf = b->ufd);
	FSEEK(bf,(OFF_T)0,SEEK_END);
	len=FTELL(bf);
	if (loc >= len || b->useek == 0)
		return(0);
#ifdef NO_TRUNCATE
	if (b->ufnm == NULL)
		return 0;
	rc = 0;
	fclose(b->ufd);
	if (!loc) {
		if (!(bf = FOPEN(b->ufnm, f__w_mode[b->ufmt])))
			rc = 1;
		if (b->uwrt)
			b->uwrt = 1;
		goto done;
		}
	if (!(bf = FOPEN(b->ufnm, f__r_mode[0]))
	 || !(tf = tmpfile())) {
#ifdef NON_UNIX_STDIO
 bad:
#endif
		rc = 1;
		goto done;
		}
	if (copy(bf, (long)loc, tf)) {
 bad1:
		rc = 1;
		goto done1;
		}
	if (!(bf = FREOPEN(b->ufnm, f__w_mode[0], bf)))
		goto bad1;
	rewind(tf);
	if (copy(tf, (long)loc, bf))
		goto bad1;
	b->uwrt = 1;
	b->urw = 2;
#ifdef NON_UNIX_STDIO
	if (b->ufmt) {
		fclose(bf);
		if (!(bf = FOPEN(b->ufnm, f__w_mode[3])))
			goto bad;
		FSEEK(bf,(OFF_T)0,SEEK_END);
		b->urw = 3;
		}
#endif
done1:
	fclose(tf);
done:
	f__cf = b->ufd = bf;
#else /* NO_TRUNCATE */
	if (b->urw & 2)
		fflush(b->ufd); /* necessary on some Linux systems */
#ifndef FTRUNCATE
#define FTRUNCATE ftruncate
#endif
	rc = FTRUNCATE(fileno(b->ufd), loc);
	/* The following FSEEK is unnecessary on some systems, */
	/* but should be harmless. */
	FSEEK(b->ufd, (OFF_T)0, SEEK_END);
#endif /* NO_TRUNCATE */
	if (rc)
		err(a->aerr,111,"endfile");
	return 0;
	}
#ifdef __cplusplus
}
#endif
