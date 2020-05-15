#pragma once

#include <util/system/defaults.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>

// simplified options parser
// No 'optional argument' (e.g. "a::" in spec.) support;
// Supports '+' switch (see opt.h), does not support '-';

/** Typical use
   Opt2 opt(argc, argv, "A:b:c", 3); <- 3 more arguments expected, opt.Pos[0], etc.
     ** Usage description for options is provided through functions that query values **
   const char *a = opt.Arg('A', "<var_name> - usage of -A"); <- This option is required
   int         b = opt.Int('b', "<var_name> - usage of -b", 2); <- This option has default value, not required
   bool        c = opt.Has('c', "- usage of -c"); <- switches are always optional

       ** Additional argument names are provided in AutoUsage call **
            ** AutoUsage generages 'USAGE' text automatically **
   if (opt.AutoUsage("<L> <M>")) <- Returns 1 if there was any error in getopt
      return 1;
   OR: opt.AutoUsageErr("<L> <M>"); <- Will terminate program for you :)
*/

// Note: struct Opt2Param can be moved to cpp-file
struct Opt2Param {
    char opt;
    bool HasArg;
    bool IsFound;
    bool IsNumeric;
    bool IsRequired;
    bool MultipleUse;
    const char* DefValue;
    TString DefValueStr;
    TString HelpUsage;
    TVector<const char*> ActualValue;
    const char* LongOptName;
    Opt2Param()
        : HasArg(false)
        , IsFound(0)
        , IsNumeric(0)
        , IsRequired(0)
        , MultipleUse(0)
        , DefValue(nullptr)
        , LongOptName(nullptr)
    {
    }
};

struct IntRange {
    int Left, Right;
    IntRange() = delete;
    IntRange(int both)
        : Left(both)
        , Right(both)
    {
    }

    IntRange(int left, int right)
        : Left(left)
        , Right(right)
    {
    }
};

class Opt2 {
public:
    Opt2() = default;

    Opt2(int argc, char* const* argv, const char* optspec, IntRange free_args_num = -1, const char* long_alias = nullptr) {
        Init(argc, argv, optspec, free_args_num, long_alias);
    }

    // Init throws exception only in case of incorrect optspec.
    // In other cases, consult HasErrors or call AutoUsage()
    void Init(int argc, char* const* argv, const char* optspec, IntRange free_args_num = -1, const char* long_alias = nullptr);

    // In case of incorrect options, constructs and prints Usage text,
    // usually to stderr (however, to stdout if '-?' switch was used), and returns 1.
    int AutoUsage(const char* free_arg_names = "");

    // same as AutoUsage but calls exit(1) instead of error code
    void AutoUsageErr(const char* free_arg_names = "");

    // For options with parameters
    const char* Arg(char opt, const char* helpUsage, const char* defValue, bool required = false);
    const char* Arg(char opt, const char* helpUsage) {
        return Arg(opt, helpUsage, nullptr, true);
    }
    const char* Arg(char opt, const char* helpUsage, TString defValue, bool required = false);

    // Options with parameters that can be specified several times
    const TVector<const char*>& MArg(char opt, const char* helpUsage);

    // Get() + strtol, may set up HasErrors
    long Int(char opt, const char* helpUsage, long defValue, bool required = false);
    long Int(char opt, const char* helpUsage) {
        return Int(opt, helpUsage, 0, true);
    }

    // Get() + strtoul, may set up HasErrors
    unsigned long UInt(char opt, const char* helpUsage, unsigned long defValue, bool required = false);
    unsigned long UInt(char opt, const char* helpUsage) {
        return UInt(opt, helpUsage, 0, true);
    }

    // For options w/o parameters
    bool Has(char opt, const char* helpUsage);

    // Add user defined error message and set error flag
    void AddError(const char* message = nullptr);

public:
    // non-option args
    TVector<char*> Pos;
    bool HasErrors;

private:
    bool BadPosCount;
    char UnknownOption;
    char* UnknownLongOption;
    char OptionMissingArg;
    char OptionWrongArg;
    char RequiredOptionMissing;
    TVector<TString> UserErrorMessages;

protected:
    int Argc;
    char* const* Argv;
    int MinArgs, MaxArgs;
    ui8 SpecsMap[256];
    TVector<Opt2Param> Specs;
    TString alias_copy;
    void EatArgv(const char* optspec, const char* long_alias);
    void Clear();
    Opt2Param& GetInternal(char opt, const char* defValue, const char* helpUsage, bool required);
};
