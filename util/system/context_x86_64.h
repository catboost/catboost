#pragma once

#define MJB_RBP 1
#define MJB_R12 2
#define MJB_R13 3
#define MJB_RSP 6
#define MJB_PC 7

#define JUMP_FUNCTION MJB_R12
#define JUMP_ARGUMENT MJB_R13

typedef long int __myjmp_buf[8];
