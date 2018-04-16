#pragma once

// implementation of posix getopt using last getopt for demonstration purposes

#include "last_getopt.h"

namespace NLastGetopt {
    extern char* optarg;
    extern int optind;
    extern int optopt;
    extern int opterr;
    extern int optreset;

    enum {
        no_argument = NO_ARGUMENT,
        required_argument = REQUIRED_ARGUMENT,
        optional_argument = OPTIONAL_ARGUMENT,
    };

    struct option {
        const char* name;
        int has_arg;
        int* flag;
        int val;
    };

    int getopt(int argc, char* const* argv, const char* optstring);
    int getopt_long(int argc, char* const* argv, const char* optstring,
                    const struct option* longopts, int* longindex);
    int getopt_long_only(int argc, char* const* argv, const char* optstring,
                         const struct option* longopts, int* longindex);
}
