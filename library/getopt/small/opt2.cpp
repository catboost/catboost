#include "opt2.h"

#include <util/generic/hash.h>
#include <util/generic/utility.h>
#include <util/generic/yexception.h>
#include <util/str_stl.h>

#include <stdio.h>
#include <errno.h>
#include <ctype.h>

void Opt2::Clear() {
    Specs.clear();
    memset(SpecsMap, 0, sizeof(SpecsMap));
    Pos.clear();
}

void Opt2::Init(int argc, char* const* argv, const char* optspec, IntRange free_args_num, const char* long_alias) {
    Clear();
    Argc = argc;
    Argv = argv;
    HasErrors = false, BadPosCount = false, UnknownOption = 0, OptionMissingArg = 0;
    UnknownLongOption = nullptr;
    OptionWrongArg = 0, RequiredOptionMissing = 0;
    EatArgv(optspec, long_alias);
    MinArgs = Min<int>(free_args_num.Left, free_args_num.Right);
    MaxArgs = Max<int>(free_args_num.Left, free_args_num.Right);
    if (!HasErrors && MinArgs != -1 && ((int)Pos.size() < MinArgs || (int)Pos.size() > MaxArgs))
        BadPosCount = HasErrors = true;
}

void Opt2::EatArgv(const char* optspec, const char* long_alias) {
    // some flags
    bool require_order = false;
    if (*optspec == '+') {
        require_order = true;
        optspec++;
    }
    if (*optspec == '-')
        ythrow yexception() << "Flag '-' can not be used in Opt2's optspec";
    // step 1 - parse optspec
    for (const char* s = optspec; *s; s++) {
        if (SpecsMap[(ui8)*s])
            ythrow yexception() << "Symbol '" << *s << "' is met twice in Opt2's optspec";
        if (*s == '?' || *s == '-')
            ythrow yexception() << "Opt2: Symbol '" << *s << "' can not be used in optspec because it is reserved";
        Specs.push_back(Opt2Param());
        SpecsMap[(ui8)*s] = (ui8)Specs.size(); // actual index + 1
        Specs.back().opt = *s;
        if (s[1] == ':') {
            Specs.back().HasArg = true;
            if (s[2] == ':')
                ythrow yexception() << "Opt2 does not accept optional parameters (e.g. \"a::\") in optspec";
            s++;
        }
    }
    // long_alias has a form "long-name1=A,long-name2=B", etc.
    // This implementation is limited to aliasing a single long option
    // with single short option (extend it if you really need).
    THashMap<const char*, char> long2short;
    long2short["help"] = '?';
    long_alias = long_alias ? long_alias : "";
    alias_copy = long_alias;
    for (char* s = alias_copy.begin(); s && *s;) {
        char* eq = strchr(s, '=');
        char* comma = strchr(s, ',');
        if (comma)
            *comma = 0;
        if (!eq || (comma && comma < eq))
            ythrow yexception() << "Opt2, long_alias: '=' is expected after " << s;
        *eq++ = 0;
        if (!*eq || eq[1])
            ythrow yexception() << "Opt2, long_alias: single letter must be assigned to " << s;
        if (!SpecsMap[(ui8)*eq])
            ythrow yexception() << "Opt2, long_alias: trying to assign unknown option '" << *eq << "' to " << s;
        Opt2Param& p = Specs[SpecsMap[(ui8)*eq] - 1];
        // If several long options aliased to some letter, only last one is shown in usage
        p.LongOptName = s;
        if (long2short.find(s) != long2short.end())
            ythrow yexception() << "Opt2, long_alias: " << s << " specified twice";
        long2short[s] = *eq;
        s = comma ? comma + 1 : nullptr;
    }

    if (Argc < 1) {
        HasErrors = true;
        return;
    }

    // step 2 - parse argv
    int ind = 1;
    for (; ind != Argc; ind++) {
        if (*Argv[ind] != '-') {
            if (require_order) // everything now goes to Pos
                break;
            Pos.push_back(Argv[ind]);
            continue;
        }
        const char* s = Argv[ind] + 1;

        if (*s == '-') {
            if (!*++s) { // `--' terminates the list of options
                ind++;
                break;
            }
            // long option always spans one argv (--switch or --option-name=value)
            const char* eq = strchr(s, '=');
            TString lname(s, eq ? (size_t)(eq - s) : (size_t)strlen(s));
            THashMap<const char*, char>::iterator i = long2short.find(lname.data());
            if (i == long2short.end()) {
                UnknownLongOption = strdup(lname.data()); // free'd in AutoUsage()
                HasErrors = true;
                return;
            }
            if (i->second == '?') {
                UnknownOption = '?';
                HasErrors = true;
                continue;
            }
            Opt2Param& p = Specs[SpecsMap[(ui8)i->second] - 1];
            p.IsFound = true;
            if (p.HasArg && !eq) {
                HasErrors = true;
                OptionMissingArg = p.opt; // short option, indeed
                return;
            }
            if (!p.HasArg && eq) {
                HasErrors = true;
                OptionWrongArg = p.opt; // short option, indeed
                return;
            }
            if (eq)
                p.ActualValue.push_back(eq + 1);
            continue;
        }

        for (; *s; s++) {
            if (!SpecsMap[(ui8)*s]) {
                UnknownOption = *s;
                HasErrors = true;
                if (*s == '?')
                    continue;
                return;
            }
            Opt2Param& p = Specs[SpecsMap[(ui8)*s] - 1];
            p.IsFound = true;
            if (p.HasArg) {
                if (s[1])
                    p.ActualValue.push_back(s + 1);
                else {
                    ind++;
                    if (ind == Argc) {
                        HasErrors = true;
                        OptionMissingArg = *s;
                        p.IsFound = false;
                        return;
                    }
                    p.ActualValue.push_back(Argv[ind]);
                }
                break;
            }
        }
    }
    for (; ind != Argc; ind++)
        Pos.push_back(Argv[ind]);
}

Opt2Param& Opt2::GetInternal(char opt, const char* defValue, const char* helpUsage, bool requred) {
    if (!SpecsMap[(ui8)opt])
        ythrow yexception() << "Unspecified option character '" << opt << "' asked from Opt2::Get";
    Opt2Param& p = Specs[SpecsMap[(ui8)opt] - 1];
    p.DefValue = defValue;
    p.HelpUsage = helpUsage;
    p.IsRequired = requred;
    if (!p.IsFound && requred && !HasErrors) {
        RequiredOptionMissing = opt;
        HasErrors = true;
    }
    return p;
}

// For options with parameters
const char* Opt2::Arg(char opt, const char* help, const char* def, bool required) {
    Opt2Param& p = GetInternal(opt, def, help, required);
    if (!p.HasArg)
        ythrow yexception() << "Opt2::Arg called for '" << opt << "' which is an option without argument";
    return p.IsFound ? p.ActualValue.empty() ? nullptr : p.ActualValue.back() : def;
}

// For options with parameters
const char* Opt2::Arg(char opt, const char* help, TString def, bool required) {
    Opt2Param& p = GetInternal(opt, nullptr, help, required);
    if (!p.HasArg)
        ythrow yexception() << "Opt2::Arg called for '" << opt << "' which is an option without argument";
    p.DefValueStr = def;
    p.DefValue = p.DefValueStr.begin();
    return p.IsFound ? p.ActualValue.empty() ? nullptr : p.ActualValue.back() : p.DefValue;
}

// Options with parameters that can be specified several times
const TVector<const char*>& Opt2::MArg(char opt, const char* help) {
    Opt2Param& p = GetInternal(opt, nullptr, help, false);
    p.MultipleUse = true;
    if (!p.HasArg)
        ythrow yexception() << "Opt2::Arg called for '" << opt << "' which is an option without argument";
    return p.ActualValue;
}

/// For options w/o parameters
bool Opt2::Has(char opt, const char* help) {
    Opt2Param& p = GetInternal(opt, nullptr, help, false);
    if (p.HasArg)
        ythrow yexception() << "Opt2::Has called for '" << opt << "' which is an option with argument";
    return p.IsFound;
}

// Get() + strtol, may set up HasErrors
long Opt2::Int(char opt, const char* help, long def, bool required) {
    Opt2Param& p = GetInternal(opt, (char*)(uintptr_t)def, help, required);
    if (!p.HasArg)
        ythrow yexception() << "Opt2::Int called for '" << opt << "' which is an option without argument";
    p.IsNumeric = true;
    if (!p.IsFound || p.ActualValue.empty() || !p.ActualValue.back())
        return def;
    char* e;
    long rv = strtol(p.ActualValue.back(), &e, 10);
    if (e == p.ActualValue.back() || *e) {
        OptionWrongArg = opt;
        HasErrors = true;
    }
    return rv;
}

// Get() + strtoul, may set up HasErrors
unsigned long Opt2::UInt(char opt, const char* help, unsigned long def, bool required) {
    Opt2Param& p = GetInternal(opt, (char*)(uintptr_t)def, help, required);
    if (!p.HasArg)
        ythrow yexception() << "Opt2::UInt called for '" << opt << "' which is an option without argument";
    p.IsNumeric = true;
    if (!p.IsFound || p.ActualValue.empty() || !p.ActualValue.back())
        return def;
    char* e;
    unsigned long rv = strtoul(p.ActualValue.back(), &e, 10);
    if (e == p.ActualValue.back() || *e) {
        OptionWrongArg = opt;
        HasErrors = true;
    }
    return rv;
}

// Add user defined error message and set error flag
void Opt2::AddError(const char* message) {
    HasErrors = true;
    if (message)
        UserErrorMessages.push_back(message);
}

int Opt2::AutoUsage(const char* free_arg_names) {
    if (!HasErrors)
        return 0;
    FILE* where = UnknownOption == '?' ? stdout : stderr;
    char req_str[256], nreq_str[256];
    int req = 0, nreq = 0;
    for (int n = 0; n < (int)Specs.size(); n++)
        if (Specs[n].IsRequired)
            req_str[req++] = Specs[n].opt;
        else
            nreq_str[nreq++] = Specs[n].opt;
    req_str[req] = 0, nreq_str[nreq] = 0;
    const char* prog = strrchr(Argv[0], LOCSLASH_C);
    prog = prog ? prog + 1 : Argv[0];
    fprintf(where, "Usage: %s%s%s%s%s%s%s%s\n", prog, req ? " -" : "", req_str,
            nreq ? " [-" : "", nreq_str, nreq ? "]" : "",
            free_arg_names && *free_arg_names ? " " : "", free_arg_names);
    for (auto& spec : Specs) {
        const char* hlp = !spec.HelpUsage.empty() ? spec.HelpUsage.data() : spec.HasArg ? "<arg>" : "";
        if (!spec.HasArg || spec.IsRequired)
            fprintf(where, "  -%c %s\n", spec.opt, hlp);
        else if (!spec.IsNumeric)
            fprintf(where, "  -%c %s [Default: %s]\n", spec.opt, hlp, spec.DefValue);
        else
            fprintf(where, "  -%c %s [Def.val: %li]\n", spec.opt, hlp, (long)(uintptr_t)spec.DefValue);
        if (spec.LongOptName)
            fprintf(where, "    --%s%s - same as -%c\n", spec.LongOptName, spec.HasArg ? "=<argument>" : "", spec.opt);
    }
    if (OptionMissingArg)
        fprintf(where, " *** Option '%c' is missing required argument\n", OptionMissingArg);
    if (OptionWrongArg)
        fprintf(where, " *** Incorrect argument for option '%c'\n", OptionWrongArg);
    if (UnknownOption && UnknownOption != '?')
        fprintf(where, " *** Unknown option '%c'\n", UnknownOption);
    if (UnknownLongOption) {
        fprintf(where, " *** Unknown long option '%s'\n", UnknownLongOption);
        free(UnknownLongOption);
        UnknownLongOption = nullptr;
    }
    if (RequiredOptionMissing)
        fprintf(where, " *** Required option '%c' missing\n", RequiredOptionMissing);
    if (BadPosCount && MinArgs != MaxArgs)
        fprintf(where, " *** %i free argument(s) supplied, expected %i to %i\n", (int)Pos.size(), MinArgs, MaxArgs);
    if (BadPosCount && MinArgs == MaxArgs)
        fprintf(where, " *** %i free argument(s) supplied, expected %i\n", (int)Pos.size(), MinArgs);
    for (const auto& userErrorMessage : UserErrorMessages)
        fprintf(where, " *** %s\n", userErrorMessage.data());
    return UnknownOption == '?' ? 1 : 2;
}

void Opt2::AutoUsageErr(const char* free_arg_names) {
    if (AutoUsage(free_arg_names))
        exit(1);
}

#ifdef OPT2_TEST
// TODO: convert it to unittest

bool opt2_ut_fail = false, opt_ut_verbose = false;
const char* ut_optspec;
int ut_real(TString args, bool err_exp, const char* A_exp, int b_exp, bool a_exp, const char* p1_exp, const char* p2_exp) {
    char* argv[32];
    int argc = sf(' ', argv, args.begin());
    Opt2 opt(argc, argv, ut_optspec, 2, "option-1=A,option-2=a,");
    const char* A = opt.Arg('A', "<qqq> - blah");
    int b = opt.Int('b', "<rrr> - blah", 2);
    bool a = opt.Has('a', "- blah");
    /*const char *C = */ opt.Arg('C', "<ccc> - blah", 0);

    if (opt_ut_verbose)
        opt.AutoUsage("");
    if (opt.HasErrors != err_exp)
        return 1;
    if (err_exp)
        return false;
    if (!A && A_exp || A && !A_exp || A && A_exp && strcmp(A, A_exp))
        return 2;
    if (b != b_exp)
        return 3;
    if (a != a_exp)
        return 4;
    if (strcmp(opt.Pos[0], p1_exp))
        return 5;
    if (strcmp(opt.Pos[1], p2_exp))
        return 6;
    return false;
}

void ut(const char* args, bool err_exp, const char* A_exp, int b_exp, bool a_exp, const char* p1_exp, const char* p2_exp) {
    if (opt_ut_verbose)
        fprintf(stderr, "Testing: %s\n", args);
    if (int rv = ut_real(args, err_exp, A_exp, b_exp, a_exp, p1_exp, p2_exp)) {
        opt2_ut_fail = true;
        fprintf(stderr, "Test %i failed for: %s\n", rv, args);
    } else {
        if (opt_ut_verbose)
            fprintf(stderr, "OK\n");
    }
}

int main(int argc, char* argv[]) {
    Opt2 opt(argc, argv, "v", 0);
    opt_ut_verbose = opt.Has('v', "- some verboseness");
    opt.AutoUsageErr("");
    ut_optspec = "A:ab:C:";
    ut("prog -A argA -a -b 22 -C argC Pos1 Pos2", false, "argA", 22, true, "Pos1", "Pos2");
    ut("prog Pos1 -A argA -a -C argC Pos2", false, "argA", 2, true, "Pos1", "Pos2");
    ut("prog -A argA Pos1 -b22 Pos2 -C argC", false, "argA", 22, false, "Pos1", "Pos2");
    ut("prog -A argA Pos1 -b 22 Pos2 -C", true, "argA", 22, false, "Pos1", "Pos2");
    ut("prog -A argA -a -b 22 -C Pos1 Pos2", true, "argA", 22, true, "Pos1", "Pos2");
    ut("prog -A argA -a -b two -C argC Pos1 Pos2", true, "argA", 2, true, "Pos1", "Pos2");
    ut("prog -a -b 22 -C argC Pos1 Pos2", true, "argA", 22, true, "Pos1", "Pos2");
    ut("prog Pos1 --option-1=argA -a -C argC Pos2", false, "argA", 2, true, "Pos1", "Pos2");
    ut("prog Pos1 -A argA --option-1 -a -C argC Pos2", true, "argA", 2, true, "Pos1", "Pos2");
    ut("prog -A argA --option-2 -b -22 -C argC Pos1 Pos2", false, "argA", -22, true, "Pos1", "Pos2");
    ut("prog -A argA --option-2 -b -22 -- -C argC", false, "argA", -22, true, "-C", "argC");
    ut("prog -A argA --option-2=1 -b -22 -C argC Pos1 Pos2", true, "argA", -22, true, "Pos1", "Pos2");

    ut_optspec = "+A:ab:C:";
    ut("prog -A argA --option-2 v1 -C", false, "argA", 2, true, "v1", "-C");
    ut("prog -A argA --option-2 v1 -C argC", true, "argA", 2, true, "v1", "-C");
    if (!opt2_ut_fail)
        fprintf(stderr, "All OK\n");
    return opt2_ut_fail;
}

#endif // OPT2_TEST
