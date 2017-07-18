#pragma once

#include <util/generic/yexception.h>

extern void (* volatile INTERRUPTED_FUNC)();

class TInterruptException : public yexception {

};

void SetInterruptHandler(void (*func)());
void ResetInterruptHandler();
void CheckInterrupted();
