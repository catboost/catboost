#pragma once

typedef unsigned long __myjmp_buf[22];

#define MJB_X19 0
#define MJB_X20 1
#define MJB_LR 11
#define MJB_PC 12

#define FRAME_CNT 10
#define PROGR_CNT MJB_PC
#define STACK_CNT 13

#define JUMP_FUNCTION MJB_X19
#define JUMP_ARGUMENT MJB_X20
#define JUMP_LINK MJB_LR
