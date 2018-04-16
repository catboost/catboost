#include "posix_getopt.h"

#include <util/generic/ptr.h>

#include <ctype.h>

namespace NLastGetopt {
    char* optarg;
    int optind;
    int optopt;
    int opterr;
    int optreset;

    static THolder<TOpts> Opts;
    static THolder<TOptsParser> OptsParser;

    int getopt_long_impl(int argc, char* const* argv, const char* optstring,
                         const struct option* longopts, int* longindex, bool long_only) {
        if (!Opts || optreset == 1) {
            optarg = nullptr;
            optind = 1;
            opterr = 1;
            optreset = 0;
            Opts.Reset(new TOpts(TOpts::Default(optstring)));

            Opts->AllowSingleDashForLong_ = long_only;

            for (const struct option* o = longopts; o != nullptr && o->name != nullptr; ++o) {
                TOpt* opt;
                if ((unsigned)o->val < 0x80 && isalnum(o->val)) {
                    opt = &Opts->CharOption(char(o->val));
                    opt->AddLongName(o->name);
                } else {
                    Opts->AddLongOption(o->name);
                    opt = const_cast<TOpt*>(&Opts->GetLongOption(o->name));
                }
                opt->HasArg_ = EHasArg(o->has_arg);
                opt->UserValue(o->flag);
            }

            OptsParser.Reset(new TOptsParser(&*Opts, argc, (const char**)argv));
        }

        optarg = nullptr;

        try {
            if (!OptsParser->Next()) {
                return -1;
            } else {
                optarg = (char*)OptsParser->CurVal();
                optind = (int)OptsParser->Pos_;
                if (longindex && OptsParser->CurOpt())
                    *longindex = (int)Opts->IndexOf(OptsParser->CurOpt());
                return OptsParser->CurOpt() ? OptsParser->CurOpt()->GetCharOr0() : 1;
            }
        } catch (const NLastGetopt::TException&) {
            return '?';
        }
    }

    int getopt_long(int argc, char* const* argv, const char* optstring,
                    const struct option* longopts, int* longindex) {
        return getopt_long_impl(argc, argv, optstring, longopts, longindex, false);
    }

    int getopt_long_only(int argc, char* const* argv, const char* optstring,
                         const struct option* longopts, int* longindex) {
        return getopt_long_impl(argc, argv, optstring, longopts, longindex, true);
    }

    // XXX: leading colon is not supported
    // XXX: updating optind by client is not supported
    int getopt(int argc, char* const* argv, const char* optstring) {
        return getopt_long(argc, argv, optstring, nullptr, nullptr);
    }

}
