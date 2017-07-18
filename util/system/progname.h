#pragma once

class TString;

void SetProgramName(const char* argv0);

#define SAVE_PROGRAM_NAME        \
    do {                         \
        SetProgramName(argv[0]); \
    } while (0)

/// guaranted return the same immutable instance of TString
const TString& GetProgramName();
