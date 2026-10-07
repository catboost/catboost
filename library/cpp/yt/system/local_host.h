#pragma once

#include <util/generic/strbuf.h>

#include <optional>
#include <string>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

/*
 * Get*Raw & Set* functions work with statically allocated buffer.
 * These functions are carefully engineered to be as safe and as robust
 * as possible, because they may be called at any time (e. g. during
 * invocation of atexit() hooks; or during static initialization).
 * They must never throw / crash / corrupt program.
 *
 * Get*Raw returns a string view pointing to a static buffer
 * whose content never changes. Get*Raw may be called concurrently,
 * it is a lock-free function.
 *
 * Set* checks if the new value differs from the latest one, and saves
 * new value to the static memory if so. New value becomes visible for future Get*s.
 *
 * Set* may be called concurrently, but it blocks.
 * It also updates the stored local YP cluster.
 */
TStringBuf GetLocalHostNameRaw() noexcept;
TStringBuf GetLocalYPClusterRaw() noexcept;

void SetLocalHostName(TStringBuf hostName) noexcept;

//! Get* wrap Get*Raw for more convenient usage.
//! The price is a (possibly) dynamically allocated string.
std::string GetLocalHostName();
std::string GetLocalYPCluster();

std::optional<TStringBuf> InferYPClusterFromHostNameRaw(TStringBuf hostName);

//! Returns an eternal copy of #hostName.
//! Equal host names are backed by the same storage.
TStringBuf InternHostName(TStringBuf hostName);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
