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

/**
 * You shouldn't ever need this, unless you are writing some DLL modules,
 * which might get unloaded before application termination (nginx's modules, for example).
 * If a DLL sets exit handlers which belong to the DLL itself, these handlers point to
 * nowhere after the DLL has been unloaded, and an attempt to invoke any of those at the
 * application termination leads to a crash.
 */
void DisableExitHandlers();
