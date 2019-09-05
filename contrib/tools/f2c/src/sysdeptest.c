/* This is never meant to be executed; we just want to check for the */
/* presence of mkdtemp and mkstemp by whether this links without error. */

#include <stdio.h>
#include <unistd.h>

 int
#ifdef KR_headers
main(argc, argv) int argc; char **argv;
#else
main(int argc, char **argv)
#endif
{
	char buf[16];
	if (argc < 0) {
#ifndef NO_MKDTEMP
		mkdtemp(buf);
#else
		mkstemp(buf);
#endif
		}
	return 0;
	}
