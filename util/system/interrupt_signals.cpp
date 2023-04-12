#include "interrupt_signals.h"

#include <util/generic/yexception.h>

#include <csignal>

static void (*InterruptSignalsHandler)(int signum) = nullptr;

#ifdef _win_

    #include <windows.h>

static BOOL WINAPI WindowsSignalsHandler(_In_ DWORD dwCtrlType) {
    if (!InterruptSignalsHandler) {
        return FALSE;
    }

    switch (dwCtrlType) {
        case CTRL_C_EVENT:
            InterruptSignalsHandler(SIGINT);
            return TRUE;
        case CTRL_BREAK_EVENT:
            InterruptSignalsHandler(SIGTERM);
            return TRUE;
        case CTRL_CLOSE_EVENT:
            InterruptSignalsHandler(SIGHUP);
            return TRUE;
        default:
            return FALSE;
    }
    Y_UNREACHABLE();
}

#endif

// separate function is to enforce 'extern "C"' linkage
extern "C" void CppSignalsHandler(int signum) {
    if (InterruptSignalsHandler) {
        InterruptSignalsHandler(signum);
    }
}

void SetInterruptSignalsHandler(void (*handler)(int signum)) {
    InterruptSignalsHandler = handler;
#ifdef _win_
    if (!SetConsoleCtrlHandler(WindowsSignalsHandler, TRUE)) {
        ythrow TSystemError() << "SetConsoleCtrlHandler failed: " << LastSystemErrorText();
    }
    for (int signum : {SIGINT, SIGTERM}) {
#else
    for (int signum : {SIGINT, SIGTERM, SIGHUP}) {
#endif
        if (std::signal(signum, CppSignalsHandler) == SIG_ERR) {
            ythrow TSystemError() << "std::signal failed to set handler for signal with id " << signum;
        }
    }
}
