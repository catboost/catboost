#pragma once

#include "private_namespace-linux.h"

#undef pthread_create_wrapper
#define tsd_boot_wrapper JEMALLOC_N(tsd_boot_wrapper)
#define tsd_init_check_recursion JEMALLOC_N(tsd_init_check_recursion)
#define tsd_init_finish JEMALLOC_N(tsd_init_finish)
#define tsd_init_head JEMALLOC_N(tsd_init_head)
#undef tsd_tls
#define zone_register JEMALLOC_N(zone_register)
