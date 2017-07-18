#include "platform.h"

#include <cstddef>

static_assert(sizeof(char) == SIZEOF_CHAR, "incorrect SIZEOF_CHAR");
static_assert(sizeof(short) == SIZEOF_SHORT, "incorrect SIZEOF_SHORT");
static_assert(sizeof(int) == SIZEOF_INT, "incorrect SIZEOF_INT");
static_assert(sizeof(long) == SIZEOF_LONG, "incorrect SIZEOF_LONG");
static_assert(sizeof(long long) == SIZEOF_LONG_LONG, "incorrect SIZEOF_LONG_LONG");

static_assert(sizeof(unsigned char) == SIZEOF_UNSIGNED_CHAR, "incorrect SIZEOF_UNSIGNED_CHAR");
static_assert(sizeof(unsigned short) == SIZEOF_UNSIGNED_SHORT, "incorrect SIZEOF_UNSIGNED_SHORT");
static_assert(sizeof(unsigned int) == SIZEOF_UNSIGNED_INT, "incorrect SIZEOF_UNSIGNED_INT");
static_assert(sizeof(unsigned long) == SIZEOF_UNSIGNED_LONG, "incorrect SIZEOF_UNSIGNED_LONG");
static_assert(sizeof(unsigned long long) == SIZEOF_UNSIGNED_LONG_LONG, "incorrect SIZEOF_UNSIGNED_LONG_LONG");

static_assert(sizeof(void*) == SIZEOF_PTR, "incorrect SIZEOF_PTR");
static_assert(sizeof(void*) == sizeof(size_t), "unsupported platform");
