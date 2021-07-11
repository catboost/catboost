#pragma once

#if defined(_x86_64_)
    #include "context_x86_64.h"
#elif defined(_i386_)
    #include "context_i686.h"
#endif

#define PROGR_CNT MJB_PC
#define STACK_CNT MJB_RSP
#define FRAME_CNT MJB_RBP
#define EXTRA_PUSH_ARGS 1
