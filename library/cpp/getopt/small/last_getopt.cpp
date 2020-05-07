#include "last_getopt.h"

namespace NLastGetopt {
    void PrintUsageAndExit(const TOptsParser* parser) {
        parser->PrintUsage();
        exit(0);
    }

}
