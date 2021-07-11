#pragma once

#include "platform.h"

#include <csignal>

#ifdef _win_
    #ifndef SIGHUP
        #define SIGHUP 1 /* Hangup (POSIX).  */
    #endif
#endif

/**
 * Set handler for interrupt signals.
 *
 * All OSes:      SIGINT, SIGTERM (defined by C++ standard)
 * UNIX variants: Also SIGHUP
 * Windows:       CTRL_C_EVENT handled as SIGINT, CTRL_BREAK_EVENT as SIGTERM, CTRL_CLOSE_EVENT as SIGHUP
 *
 * \param handler                       Signal handler to use. Pass nullptr to clear currently set handler.
 */
void SetInterruptSignalsHandler(void (*handler)(int signum));
