#pragma once

#include <util/system/defaults.h>

ui32 GetRXPacketsLostCounter(int sock) noexcept;
i32 GetNumRXPacketsLost(int sock, ui32 initialValue) noexcept;
void EnableTXPacketsCounter(int sock) noexcept;
