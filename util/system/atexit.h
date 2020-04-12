#pragma once

#include "defaults.h"

using TAtExitFunc = void (*)(void*);
using TTraditionalAtExitFunc = void (*)();

void AtExit(TAtExitFunc func, void* ctx);
void AtExit(TAtExitFunc func, void* ctx, size_t priority);

void AtExit(TTraditionalAtExitFunc func);
void AtExit(TTraditionalAtExitFunc func, size_t priority);

bool ExitStarted();

/**
 * Generally it's a bad idea to call this method except for some rare cases,
 * like graceful python DLL module unload.
 * This function is not threadsafe.
 * Calls in the moment when application is not terminating - bad idea.
 */
void ManualRunAtExitFinalizers();
