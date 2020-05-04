#pragma once

#include "modchooser.h"

namespace NLastGetopt {
    /// Create an option that generates completion.
    TOpt MakeCompletionOpt(const TOpts* opts, TString command, TString optName = "completion");

    /// Create a mode that generates completion.
    THolder<TMainClassArgs> MakeCompletionMod(const TModChooser* modChooser, TString command, TString modName = "completion");
}
