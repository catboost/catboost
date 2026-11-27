#pragma once

#include <util/generic/array_ref.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>

namespace NStage0Pycc {

    // Arguments exclude argv[0]. Ya command files contain one literal argument
    // per nonempty line; their contents are not expanded recursively.
    // Inline command-file markers are ignored. The result owns its strings.
    TVector<TString> ReadCommandFileArgs(TConstArrayRef<const char*> arguments);

} // namespace NStage0Pycc
