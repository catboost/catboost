#pragma once

#include <util/generic/fwd.h>

// NOTE: This function has rare sporadic failures (throws exceptions) on FreeBSD. See REVIEW:54297
const TString& GetExecPath();

/**
 * Get openable path to the binary currently being executed.
 *
 * The path does not match the original binary location, but stays openable even
 * if the binary was moved or removed.
 *
 * On UNIX variants, utilizes the /proc FS. On Windows, equivalent to
 * GetExecPath.
 */
const TString& GetPersistentExecPath();
