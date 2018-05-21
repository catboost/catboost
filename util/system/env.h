#pragma once

#include <util/generic/string.h>

/**
 * Search the environment list provided by the host environment for associated variable.
 *
 * @param key   String identifying the name of the environmental variable to look for
 * @param def   String that returns if environmental variable not found by key
 *
 * @return      String that is associated with the matched environment variable or empty string if
 *              such variable is missing.
 *
 * @note        Use it only in pair with `SetEnv` as there may be inconsistency in their behaviour
 *              otherwise.
 * @note        Calls to `GetEnv` and `SetEnv` from different threads must be synchronized.
 * @see         SetEnv
 */
TString GetEnv(const TString& key, const TString& def = TString());

/**
 * Add or change environment variable provided by the host environment.
 *
 * @key         String identifying the name of the environment variable to set or change
 * @value       Value to assign

 * @note        Use it only in pair with `GetEnv` as there may be inconsistency in their behaviour
 *              otherwise.
 * @note        Calls to `GetEnv` and `SetEnv` from different threads must be synchronized.
 * @see         GetEnv
 */
void SetEnv(const TString& key, const TString& value);
