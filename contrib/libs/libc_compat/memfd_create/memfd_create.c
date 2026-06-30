#define _GNU_SOURCE 1
#include <sys/mman.h>
#include "syscall.h"
#include <linux/unistd.h>

int memfd_create(const char *name, unsigned flags)
{
	return syscall(__NR_memfd_create, name, flags);
}
