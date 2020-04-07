#pragma once

class IOutputStream;

namespace NColorizer {
    class TColors;

    TColors& StdErr();
    TColors& StdOut();
    TColors& AutoColors(IOutputStream&);
}
