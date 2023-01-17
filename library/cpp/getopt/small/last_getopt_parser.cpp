#include "last_getopt_parser.h"

#include <library/cpp/colorizer/colors.h>

#include <util/string/escape.h>

namespace NLastGetopt {
    void TOptsParser::Init(const TOpts* opts, int argc, const char* argv[]) {
        opts->Validate();

        Opts_ = opts;

        if (argc < 1)
            throw TUsageException() << "argv must have at least one argument";

        Argc_ = argc;
        Argv_ = argv;

        ProgramName_ = argv[0];

        Pos_ = 1;
        Sop_ = 0;
        CurrentOpt_ = nullptr;
        CurrentValue_ = nullptr;
        GotMinusMinus_ = false;
        Stopped_ = false;
        OptsSeen_.clear();
        OptsDefault_.clear();
    }

    void TOptsParser::Init(const TOpts* opts, int argc, char* argv[]) {
        Init(opts, argc, const_cast<const char**>(argv));
    }

    void TOptsParser::Swap(TOptsParser& that) {
        DoSwap(Opts_, that.Opts_);
        DoSwap(Argc_, that.Argc_);
        DoSwap(Argv_, that.Argv_);
        DoSwap(TempCurrentOpt_, that.TempCurrentOpt_);
        DoSwap(ProgramName_, that.ProgramName_);
        DoSwap(Pos_, that.Pos_);
        DoSwap(Sop_, that.Sop_);
        DoSwap(Stopped_, that.Stopped_);
        DoSwap(CurrentOpt_, that.CurrentOpt_);
        DoSwap(CurrentValue_, that.CurrentValue_);
        DoSwap(GotMinusMinus_, that.GotMinusMinus_);
        DoSwap(OptsSeen_, that.OptsSeen_);
    }

    bool TOptsParser::Commit(const TOpt* currentOpt, const TStringBuf& currentValue, size_t pos, size_t sop) {
        Pos_ = pos;
        Sop_ = sop;
        CurrentOpt_ = currentOpt;
        CurrentValue_ = currentValue;
        if (nullptr != currentOpt)
            OptsSeen_.insert(currentOpt);
        return true;
    }

    bool TOptsParser::CommitEndOfOptions(size_t pos) {
        Pos_ = pos;
        Sop_ = 0;
        Y_ASSERT(!CurOpt());
        Y_ASSERT(!CurVal());

        Y_ASSERT(!Stopped_);

        if (Opts_->FreeArgsMin_ == Opts_->FreeArgsMax_ && Argc_ - Pos_ != Opts_->FreeArgsMin_)
            throw TUsageException() << "required exactly " << Opts_->FreeArgsMin_ << " free args";
        else if (Argc_ - Pos_ < Opts_->FreeArgsMin_)
            throw TUsageException() << "required at least " << Opts_->FreeArgsMin_ << " free args";
        else if (Argc_ - Pos_ > Opts_->FreeArgsMax_)
            throw TUsageException() << "required at most " << Opts_->FreeArgsMax_ << " free args";

        return false;
    }

    bool TOptsParser::ParseUnknownShortOptWithinArg(size_t pos, size_t sop) {
        Y_ASSERT(pos < Argc_);
        const TStringBuf arg(Argv_[pos]);
        Y_ASSERT(sop > 0);
        Y_ASSERT(sop < arg.length());
        Y_ASSERT(EIO_NONE != IsOpt(arg));

        if (!Opts_->AllowUnknownCharOptions_)
            throw TUsageException() << "unknown option '" << EscapeC(arg[sop])
                                     << "' in '" << arg << "'";

        TempCurrentOpt_.Reset(new TOpt);
        TempCurrentOpt_->AddShortName(arg[sop]);

        sop += 1;

        // mimic behavior of Opt: unknown option has arg only if char is last within arg
        if (sop < arg.length()) {
            return Commit(TempCurrentOpt_.Get(), nullptr, pos, sop);
        }

        pos += 1;
        sop = 0;
        if (pos == Argc_ || EIO_NONE != IsOpt(Argv_[pos])) {
            return Commit(TempCurrentOpt_.Get(), nullptr, pos, 0);
        }

        return Commit(TempCurrentOpt_.Get(), Argv_[pos], pos + 1, 0);
    }

    bool TOptsParser::ParseShortOptWithinArg(size_t pos, size_t sop) {
        Y_ASSERT(pos < Argc_);
        const TStringBuf arg(Argv_[pos]);
        Y_ASSERT(sop > 0);
        Y_ASSERT(sop < arg.length());
        Y_ASSERT(EIO_NONE != IsOpt(arg));

        size_t p = sop;
        char c = arg[p];
        const TOpt* opt = Opts_->FindCharOption(c);
        if (!opt)
            return ParseUnknownShortOptWithinArg(pos, sop);
        p += 1;
        if (p == arg.length()) {
            return ParseOptParam(opt, pos + 1);
        }
        if (opt->GetHasArg() == NO_ARGUMENT) {
            return Commit(opt, nullptr, pos, p);
        }
        return Commit(opt, arg.SubStr(p), pos + 1, 0);
    }

    bool TOptsParser::ParseShortOptArg(size_t pos) {
        Y_ASSERT(pos < Argc_);
        const TStringBuf arg(Argv_[pos]);
        Y_ASSERT(EIO_NONE != IsOpt(arg));
        Y_ASSERT(!arg.StartsWith("--"));
        return ParseShortOptWithinArg(pos, 1);
    }

    bool TOptsParser::ParseOptArg(size_t pos) {
        Y_ASSERT(pos < Argc_);
        TStringBuf arg(Argv_[pos]);
        const EIsOpt eio = IsOpt(arg);
        Y_ASSERT(EIO_NONE != eio);
        if (EIO_DDASH == eio || EIO_PLUS == eio || (Opts_->AllowSingleDashForLong_ || !Opts_->HasAnyShortOption())) {
            // long option
            bool singleCharPrefix = EIO_DDASH != eio;
            arg.Skip(singleCharPrefix ? 1 : 2);
            TStringBuf optionName = arg.NextTok('=');
            const TOpt* option = Opts_->FindLongOption(optionName);
            if (!option) {
                if (singleCharPrefix && !arg.IsInited()) {
                    return ParseShortOptArg(pos);
                } else if (Opts_->AllowUnknownLongOptions_) {
                    return false;
                } else {
                    throw TUsageException() << "unknown option '" << optionName
                                             << "' in '" << Argv_[pos] << "'";
                }
            }
            if (arg.IsInited()) {
                if (option->GetHasArg() == NO_ARGUMENT)
                    throw TUsageException() << "option " << optionName << " must have no arg";
                return Commit(option, arg, pos + 1, 0);
            }
            ++pos;
            return ParseOptParam(option, pos);
        } else {
            return ParseShortOptArg(pos);
        }
    }

    bool TOptsParser::ParseOptParam(const TOpt* opt, size_t pos) {
        Y_ASSERT(opt);
        if (opt->GetHasArg() == NO_ARGUMENT || opt->IsEqParseOnly()) {
            return Commit(opt, nullptr, pos, 0);
        }
        if (pos == Argc_) {
            if (opt->GetHasArg() == REQUIRED_ARGUMENT)
                throw TUsageException() << "option " << opt->ToShortString() << " must have arg";
            return Commit(opt, nullptr, pos, 0);
        }
        const TStringBuf arg(Argv_[pos]);
        if (!arg.StartsWith('-') || opt->GetHasArg() == REQUIRED_ARGUMENT) {
            return Commit(opt, arg, pos + 1, 0);
        }
        return Commit(opt, nullptr, pos, 0);
    }

    TOptsParser::EIsOpt TOptsParser::IsOpt(const TStringBuf& arg) const {
        EIsOpt eio = EIO_NONE;
        if (1 < arg.length()) {
            switch (arg[0]) {
                default:
                    break;
                case '-':
                    if ('-' != arg[1])
                        eio = EIO_SDASH;
                    else if (2 < arg.length())
                        eio = EIO_DDASH;
                    break;
                case '+':
                    if (Opts_->AllowPlusForLong_)
                        eio = EIO_PLUS;
                    break;
            }
        }
        return eio;
    }

    static void memrotate(void* ptr, size_t size, size_t shift) {
        TTempBuf buf(shift);
        memcpy(buf.Data(), (char*)ptr + size - shift, shift);
        memmove((char*)ptr + shift, ptr, size - shift);
        memcpy(ptr, buf.Data(), shift);
    }

    bool TOptsParser::ParseWithPermutation() {
        Y_ASSERT(Sop_ == 0);
        Y_ASSERT(Opts_->ArgPermutation_ == PERMUTE);

        const size_t p0 = Pos_;

        size_t pc = Pos_;

        for (; pc < Argc_ && EIO_NONE == IsOpt(Argv_[pc]); ++pc) {
            // count non-args
        }

        if (pc == Argc_) {
            return CommitEndOfOptions(Pos_);
        }

        Pos_ = pc;

        bool r = ParseOptArg(Pos_);
        Y_ASSERT(r);
        while (Pos_ == pc) {
            Y_ASSERT(Sop_ > 0);
            r = ParseShortOptWithinArg(Pos_, Sop_);
            Y_ASSERT(r);
        }

        size_t p2 = Pos_;

        Y_ASSERT(p2 - pc >= 1);
        Y_ASSERT(p2 - pc <= 2);

        memrotate(Argv_ + p0, (p2 - p0) * sizeof(void*), (p2 - pc) * sizeof(void*));

        bool r2 = ParseOptArg(p0);
        Y_ASSERT(r2);
        return r2;
    }

    bool TOptsParser::DoNext() {
        Y_ASSERT(Pos_ <= Argc_);

        if (Pos_ == Argc_)
            return CommitEndOfOptions(Pos_);

        if (GotMinusMinus_ && Opts_->ArgPermutation_ == RETURN_IN_ORDER) {
            Y_ASSERT(Sop_ == 0);
            return Commit(nullptr, Argv_[Pos_], Pos_ + 1, 0);
        }

        if (Sop_ > 0)
            return ParseShortOptWithinArg(Pos_, Sop_);

        size_t pos = Pos_;
        const TStringBuf arg(Argv_[pos]);
        if (EIO_NONE != IsOpt(arg)) {
            return ParseOptArg(pos);
        } else if (arg == "--") {
            if (Opts_->ArgPermutation_ == RETURN_IN_ORDER) {
                pos += 1;
                if (pos == Argc_)
                    return CommitEndOfOptions(pos);
                GotMinusMinus_ = true;
                return Commit(nullptr, Argv_[pos], pos + 1, 0);
            } else {
                return CommitEndOfOptions(pos + 1);
            }
        } else if (Opts_->ArgPermutation_ == RETURN_IN_ORDER) {
            return Commit(nullptr, arg, pos + 1, 0);
        } else if (Opts_->ArgPermutation_ == REQUIRE_ORDER) {
            return CommitEndOfOptions(Pos_);
        } else {
            return ParseWithPermutation();
        }
    }

    bool TOptsParser::Next() {
        bool r = false;

        if (OptsDefault_.empty()) {
            CurrentOpt_ = nullptr;
            TempCurrentOpt_.Destroy();

            CurrentValue_ = nullptr;

            if (Stopped_)
                return false;

            TOptsParser copy = *this;

            r = copy.DoNext();

            Swap(copy);

            if (!r) {
                Stopped_ = true;
                // we are done; check for missing options
                Finish();
            }
        }

        if (!r && !OptsDefault_.empty()) {
            CurrentOpt_ = OptsDefault_.front();
            CurrentValue_ = CurrentOpt_->GetDefaultValue();
            OptsDefault_.pop_front();
            r = true;
        }

        if (r) {
            if (CurOpt())
                CurOpt()->FireHandlers(this);
        }

        return r;
    }

    void TOptsParser::Finish() {
        const TOpts::TOptsVector& optvec = Opts_->Opts_;
        if (optvec.size() == OptsSeen_.size())
            return;

        TVector<TString> missingLong;
        TVector<char> missingShort;

        TOpts::TOptsVector::const_iterator it;
        for (it = optvec.begin(); it != optvec.end(); ++it) {
            const TOpt* opt = (*it).Get();
            if (nullptr == opt)
                continue;
            if (OptsSeen_.contains(opt))
                continue;

            if (opt->IsRequired()) {
                const TOpt::TLongNames& optnames = opt->GetLongNames();
                if (!optnames.empty())
                    missingLong.push_back(optnames[0]);
                else {
                    const char ch = opt->GetCharOr0();
                    if (0 != ch)
                        missingShort.push_back(ch);
                }
                continue;
            }

            if (opt->HasDefaultValue())
                OptsDefault_.push_back(opt);
        }

        // also indicates subsequent options, if any, haven't been seen actually
        OptsSeen_.clear();

        const size_t nmissing = missingLong.size() + missingShort.size();
        if (0 == nmissing)
            return;

        TUsageException usage;
        usage << "The following option";
        usage << ((1 == nmissing) ? " is" : "s are");
        usage << " required:";
        for (size_t i = 0; i != missingLong.size(); ++i)
            usage << " --" << missingLong[i];
        for (size_t i = 0; i != missingShort.size(); ++i)
            usage << " -" << missingShort[i];
        throw usage; // don't need lineinfo, just the message
    }

    void TOptsParser::PrintUsage(IOutputStream& os, const NColorizer::TColors& colors) const {
        Opts_->PrintUsage(ProgramName(), os, colors);
    }

    void TOptsParser::PrintUsage(IOutputStream& os) const {
        PrintUsage(os, NColorizer::AutoColors(os));
    }

}
