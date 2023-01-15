#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define FROM_IMPL
#include "svnversion.h"

#include <util/generic/strbuf.h>

extern "C" void PrintProgramSvnVersion() {
    puts(GetProgramSvnVersion());
}

extern "C" void PrintSvnVersionAndExit0() {
    PrintProgramSvnVersion();
    exit(0);
}

extern "C" void PrintSvnVersionAndExitEx(int argc, char* argv[], const char* opts) {
    if (2 == argc) {
        for (TStringBuf all = opts, versionOpt; all.NextTok(';', versionOpt);) {
            if (versionOpt == argv[1]) {
                PrintSvnVersionAndExit0();
            }
        }
    }
}

extern "C" void PrintSvnVersionAndExit(int argc, char* argv[]) {
    PrintSvnVersionAndExitEx(argc, argv, "--version");
}

#undef FROM_IMPL
