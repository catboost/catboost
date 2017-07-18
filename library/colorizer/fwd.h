#pragma once

class TOutputStream;

namespace NColorizer {
    class TColors;

    TColors& StdErr();
    TColors& StdOut();
    TColors& AutoColors(TOutputStream&);
} // namespace NColorizer
