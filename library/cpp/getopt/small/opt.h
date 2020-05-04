#pragma once

#include "last_getopt.h"

#include <util/generic/ptr.h>
#include <util/generic/noncopyable.h>

// implementation of Opt class using last getopt

/*
 short-options syntax:

 opt-letter ::=
     [^: ]

 opt-string ::=
     '+'|'-'?({opt-letter}':'{0,2})*

 example: "AbCx:y:z::"
    {A,b,C} options without argument
    {x,y}   options with argument
    {z} option  with optional argument

 1. shortopts begins with '-'   :=> RETURN_IN_ORDER
    == non-option forces getopt to return 1 and to place non-option into optarg

 2. shortopts begins with '+'   :=> REQUIRE_ORDER
    GetEnv(_POSIX_OPTION_ORDER) :=> REQUIRE_ORDER
    == 1st non-option forces getopt to return EOF

 3. default            :=> PERMUTE
    == exchange options with non-options and place all options first

 4. '--' command line argument forces getopt to stop parsing and to return EOF
     in any case

  long options should begin by '+' sign
  or when (_getopt_long_only = 1) by '-' sign

  struct option {
   char *name : option name
   int has_arg: 0 | 1 | 2 = without | with | optional argument
   int *flag  : if (flag != 0) then getopt returns 0 and stores val into *flag
   int val    : if (flag == 0) then getopt returns val
  }

  Example:

  struct option my_opts[] = {
    { "delete", 0, &deletion_flag, DEL }, -- returns 0, deletion_flag := DEL
    { "add",    1,       NULL, 'a' }, -- returns 'a', argument in optarg
    { NULL }
  }
*/

#define OPT_RETURN_IN_ORDER "-"
#define OPT_REQUIRE_ORDER "+"
#define OPT_DONT_STORE_ARG ((void*)0)

class Opt : TNonCopyable {
public:
    enum HasArg { WithoutArg,
                  WithArg,
                  PossibleArg };

    struct Ion {
        const char* name;
        HasArg has_arg;
        int* flag;
        int val;
    };

private:
    THolder<NLastGetopt::TOpts> Opts_;
    THolder<NLastGetopt::TOptsParser> OptsParser_;
    const Ion* Ions_;
    bool GotError_;

    void Init(int argc, char* argv[], const char* optString, const Ion* longOptions = nullptr, bool longOnly = false, bool isOpen = false);

public:
    Opt(int argc, char* argv[], const char* optString, const Ion* longOptions = nullptr, bool longOnly = false, bool isOpen = false);
    Opt(int argc, const char* argv[], const char* optString, const Ion* longOptions = nullptr, bool longOnly = false, bool isOpen = false);

    // Get() means next
    int Get();
    int Get(int* longOptionIndex);
    int operator()() {
        return Get();
    }

    const char* GetArg() const {
        return Arg;
    }

    TVector<TString> GetFreeArgs() const {
        return NLastGetopt::TOptsParseResult(&*Opts_, GetArgC(), GetArgV()).GetFreeArgs();
    }

    // obsolete, use GetArg() instead
    char* Arg; /* option argument if any or NULL */

    int Ind;  /* command line index */
    bool Err; /* flag to print error messages */

    int GetArgC() const;
    const char** GetArgV() const;

    void DummyHelp(IOutputStream& os = Cerr);
};

// call before getopt. returns non-negative int, removing it from arguments (not found: -1)
// Example: returns 11 for "progname -11abc", -1 for "progname -a11"
int opt_get_number(int& argc, char* argv[]);

#define OPTION_HANDLING_PROLOG                \
    {                                         \
        int optlet;                           \
        while (EOF != (optlet = opt.Get())) { \
            switch (optlet) {
#define OPTION_HANDLING_PROLOG_ANON(S)        \
    {                                         \
        Opt opt(argc, argv, (S));             \
        int optlet;                           \
        while (EOF != (optlet = opt.Get())) { \
            switch (optlet) {
#define OPTION_HANDLE_BEGIN(opt) case opt: {
#define OPTION_HANDLE_END \
    }                     \
    break;

#define OPTION_HANDLE(opt, handle) \
    OPTION_HANDLE_BEGIN(opt)       \
    handle;                        \
    OPTION_HANDLE_END

#define OPTION_HANDLING_EPILOG                   \
    default:                                     \
        ythrow yexception() << "unknown optlet"; \
        }                                        \
        }                                        \
        }
