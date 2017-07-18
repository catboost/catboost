#pragma once

// Including this file is possible without modifying PEERDIR (for debug purposes).
// The latter is allowed only locally, so this file is named
// in such a way that including it prevents from committing the #include via ARC-1205.

#define DBGDUMP_INLINE_IF_INCLUDED inline

#include "dump.cpp"
#include "dumpers.cpp"
#include "engine.cpp"

#include <library/colorizer/colors.cpp>
#include <library/colorizer/output.cpp>

#undef DBGDUMP_INLINE_IF_INCLUDED
