#pragma once

#include <util/generic/fwd.h>
#include <util/generic/string.h>
#include <util/generic/strbuf.h>

#include <functional>

// NOTE: On Windows, functions from this header may have unexpected behavior when used with codepages that don't support all Unicode characters.
//       For more info, see https://blog.orange.tw/posts/2025-01-worstfit-unveiling-hidden-transformers-in-windows-ansi/

/**
 * Search the environment list provided by the host environment for associated variable.
 *
 * @param key   String identifying the name of the environmental variable to look for
 * @param def   String that returns if environmental variable not found by key
 *
 * @return      String that is associated with the matched environment variable or the value of `def` parameter if
 *              such variable is missing.
 *
 * @note        Use it only in pair with `SetEnv` as there may be inconsistency in their behaviour
 *              otherwise.
 * @note        Calls to `GetEnv` and environment modifying functions (like `SetEnv`) from different threads must be synchronized.
 * @see         SetEnv
 */
TString GetEnv(const TString& key, const TString& def = TString());

/**
 * Search the environment list provided by the host environment for associated variable.
 *
 * @param key   String identifying the name of the environmental variable to look for
 *
 * @return      String that is associated with the matched environment
 *              variable or empty optional value if such variable is missing.
 *
 * @note        Use it only in pair with `SetEnv` as there may be inconsistency in their behaviour otherwise.
 * @note        Calls to `TryGetEnv` and environment modifying functions (like `SetEnv`) from different threads must be synchronized.
 * @see         SetEnv
 */
TMaybe<TString> TryGetEnv(const TString& key);

/**
 * Add or change environment variable provided by the host environment.
 *
 * @param key               String identifying the name of the environment variable to set or change
 * @param value             Value to assign
 *
 * @throws TSystemError     On error
 *
 * @note        Use it only in pair with `GetEnv` as there may be inconsistency in their behaviour
 *              otherwise.
 * @note        Calls to `SetEnv` and all other env-related functions from different threads must be synchronized.
 * @see         GetEnv
 */
void SetEnv(const TString& key, const TString& value);

/**
 * Remove environment variable from the host environment.
 *
 * @param key               String identifying the name of the environment variable to remove
 *
 * @throws TSystemError     On error
 *
 * @note                    If key does not exist in the environment, then the environment is unchanged,
 *                          and the function returns normally.
 * @note                    Calls to `UnsetEnv` and and all other env-related functions from different threads must be synchronized.
 * @see                     GetEnv
 */
void UnsetEnv(const TString& key);

/**
 * Clear the host environment.
 *
 * @throws TSystemError     On error
 *
 * @note                    Calls to `ClearEnv` and all other env-related functions from different threads must be synchronized.
 */
void ClearEnv();

/**
 * Call the provided function `f` with name and value as arguments for each environment variable.
 *
 * @param f                         The function to call
 * @param ignoreMalformedStrings    If false, any malformed string in host environment (one that doesn't contain a '=') will cause an exception.
 *                                  If true, such strings will be skipped.
 *
 * @throws TSystemError         On error
 * @throws yexception           On encountering a malformed string (see `ignoreMalformedStrings`)
 *
 * @note                        The function `f` may be called with the same variable name multiple times.
 *                              Lifetime for contents of `name` and `value` is not guaranteed to last beyond the scope of the invocation of `f`.
 *
 * @note                        Calls to `IterateEnv` and environment modifying functions (like `SetEnv`) from different threads must be synchronized.
 *                              Also, `f` must not call any environment modifying functions.
 */
void IterateEnv(const std::function<void(TStringBuf /*name*/, TStringBuf /*value*/)>& f, bool ignoreMalformedStrings = false);
