#pragma once

#define MJB_SI 1
#define MJB_DI 2
#define MJB_BP 3
#define MJB_SP 4
#define MJB_PC 5

#define MJB_RBP MJB_BP
#define MJB_RSP MJB_SP

#define JUMP_FUNCTION MJB_SI
#define JUMP_ARGUMENT MJB_DI

typedef int __myjmp_buf[6];
