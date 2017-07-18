#include "interrupt.h"

void (* volatile INTERRUPTED_FUNC)() = nullptr;

void SetInterruptHandler(void (*func)()) {
    INTERRUPTED_FUNC = func;
}

void ResetInterruptHandler() {
    INTERRUPTED_FUNC = nullptr;
}

void CheckInterrupted() {
    if (INTERRUPTED_FUNC != nullptr){
        INTERRUPTED_FUNC();
    }
};
