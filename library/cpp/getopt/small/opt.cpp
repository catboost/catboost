#include "opt.h"

#include <util/system/progname.h>

#include <ctype.h>

using namespace NLastGetopt;

namespace {
    struct TOptsNoDefault: public TOpts {
        TOptsNoDefault(const TStringBuf& optstring = TStringBuf())
            : TOpts(optstring)
        {
        }
    };

}

void Opt::Init(int argc, char* argv[], const char* optString, const Ion* longOptions, bool longOnly, bool isOpen) {
    Ions_ = longOptions;
    Err = true;
    GotError_ = false;
    Ind = argc;

    Opts_.Reset(new TOptsNoDefault(optString));
    for (const Ion* o = longOptions; o != nullptr && o->name != nullptr; ++o) {
        TOpt* opt;
        if ((unsigned)o->val < 0x80 && isalnum(o->val)) {
            opt = &Opts_->CharOption(char(o->val));
            opt->AddLongName(o->name);
        } else {
            Opts_->AddLongOption(o->name);
            opt = const_cast<TOpt*>(&Opts_->GetLongOption(o->name));
        }
        opt->HasArg_ = EHasArg(o->has_arg);
        opt->UserValue(o);
    }
    Opts_->AllowSingleDashForLong_ = longOnly;
    Opts_->AllowPlusForLong_ = true;
    Opts_->AllowUnknownCharOptions_ = isOpen;
    Opts_->AllowUnknownLongOptions_ = false;

    OptsParser_.Reset(new TOptsParser(Opts_.Get(), argc, argv));
}

Opt::Opt(int argc, char* argv[], const char* optString, const Ion* longOptions, bool longOnly, bool isOpen) {
    Init(argc, argv, optString, longOptions, longOnly, isOpen);
}

Opt::Opt(int argc, const char* argv[], const char* optString, const Ion* longOptions, bool longOnly, bool isOpen) {
    Init(argc, (char**)argv, optString, longOptions, longOnly, isOpen);
}

int Opt::Get() {
    return Get(nullptr);
}

int Opt::Get(int* longOptionIndex) {
    if (GotError_)
        return EOF;

    Arg = nullptr;

    try {
        bool r = OptsParser_->Next();
        Ind = (int)OptsParser_->Pos_;
        if (!r) {
            return EOF;
        } else {
            Arg = (char*)OptsParser_->CurVal();
            if (!OptsParser_->CurOpt()) {
                // possible if RETURN_IN_ORDER
                return 1;
            } else {
                const Ion* ion = (const Ion*)OptsParser_->CurOpt()->UserValue();
                if (longOptionIndex) {
                    *longOptionIndex = int(ion - Ions_);
                }
                char c = OptsParser_->CurOpt()->GetCharOr0();
                return c != 0 ? c : ion->val;
            }
        }
    } catch (const NLastGetopt::TException&) {
        GotError_ = true;
        if (Err)
            Cerr << CurrentExceptionMessage() << Endl;
        return '?';
    }
}

void Opt::DummyHelp(IOutputStream& os) {
    Opts_->PrintUsage(GetProgramName(), os);
}

int Opt::GetArgC() const {
    return (int)OptsParser_->Argc_;
}

const char** Opt::GetArgV() const {
    return OptsParser_->Argv_;
}

int opt_get_number(int& argc, char* argv[]) {
    int num = -1;
    for (int a = 1; a < argc; a++) {
        if (*argv[a] == '-' && isdigit((ui8)argv[a][1])) {
            char* ne;
            num = strtol(argv[a] + 1, &ne, 10);
            if (*ne) {
                memmove(argv[a] + 1, ne, strlen(ne) + 1);
            } else {
                for (argc--; a < argc; a++)
                    argv[a] = argv[a + 1];
            }
            break;
        }
    }
    return num;
}
