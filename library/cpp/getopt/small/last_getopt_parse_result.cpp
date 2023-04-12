#include "last_getopt_parse_result.h"

namespace NLastGetopt {
    const TOptParseResult* TOptsParseResult::FindParseResult(const TdVec& vec, const TOpt* opt) {
        for (const auto& r : vec) {
            if (r.OptPtr() == opt)
                return &r;
        }
        return nullptr;
    }

    const TOptParseResult* TOptsParseResult::FindOptParseResult(const TOpt* opt, bool includeDefault) const {
        const TOptParseResult* r = FindParseResult(Opts_, opt);
        if (nullptr == r && includeDefault)
            r = FindParseResult(OptsDef_, opt);
        return r;
    }

    const TOptParseResult* TOptsParseResult::FindLongOptParseResult(const TString& name, bool includeDefault) const {
        return FindOptParseResult(&Parser_->Opts_->GetLongOption(name), includeDefault);
    }

    const TOptParseResult* TOptsParseResult::FindCharOptParseResult(char c, bool includeDefault) const {
        return FindOptParseResult(&Parser_->Opts_->GetCharOption(c), includeDefault);
    }

    bool TOptsParseResult::Has(const TOpt* opt, bool includeDefault) const {
        Y_ASSERT(opt);
        return FindOptParseResult(opt, includeDefault) != nullptr;
    }

    bool TOptsParseResult::Has(const TString& name, bool includeDefault) const {
        return FindLongOptParseResult(name, includeDefault) != nullptr;
    }

    bool TOptsParseResult::Has(char c, bool includeDefault) const {
        return FindCharOptParseResult(c, includeDefault) != nullptr;
    }

    const char* TOptsParseResult::Get(const TOpt* opt, bool includeDefault) const {
        Y_ASSERT(opt);
        const TOptParseResult* r = FindOptParseResult(opt, includeDefault);
        if (!r || r->Empty()) {
            try {
                throw TUsageException() << "option " << opt->ToShortString() << " is unspecified";
            } catch (...) {
                HandleError();
                // unreachable
                throw;
            }
        } else {
            return r->Back();
        }
    }

    const char* TOptsParseResult::GetOrElse(const TOpt* opt, const char* defaultValue) const {
        Y_ASSERT(opt);
        const TOptParseResult* r = FindOptParseResult(opt);
        if (!r || r->Empty()) {
            return defaultValue;
        } else {
            return r->Back();
        }
    }

    const char* TOptsParseResult::Get(const TString& name, bool includeDefault) const {
        return Get(&Parser_->Opts_->GetLongOption(name), includeDefault);
    }

    const char* TOptsParseResult::Get(char c, bool includeDefault) const {
        return Get(&Parser_->Opts_->GetCharOption(c), includeDefault);
    }

    const char* TOptsParseResult::GetOrElse(const TString& name, const char* defaultValue) const {
        if (!Has(name))
            return defaultValue;
        return Get(name);
    }

    const char* TOptsParseResult::GetOrElse(char c, const char* defaultValue) const {
        if (!Has(c))
            return defaultValue;
        return Get(c);
    }

    TOptParseResult& TOptsParseResult::OptParseResult() {
        const TOpt* opt = Parser_->CurOpt();
        Y_ASSERT(opt);
        TdVec& opts = Parser_->IsExplicit() ? Opts_ : OptsDef_;
        if (Parser_->IsExplicit()) // default options won't appear twice
            for (auto& it : opts)
                if (it.OptPtr() == opt)
                    return it;
        opts.push_back(TOptParseResult(opt));
        return opts.back();
    }

    TString TOptsParseResult::GetProgramName() const {
        return Parser_->ProgramName_;
    }

    void TOptsParseResult::PrintUsage(IOutputStream& os) const {
        Parser_->Opts_->PrintUsage(Parser_->ProgramName_, os);
    }

    size_t TOptsParseResult::GetFreeArgsPos() const {
        return Parser_->Pos_;
    }

    TVector<TString> TOptsParseResult::GetFreeArgs() const {
        TVector<TString> v;
        for (size_t i = GetFreeArgsPos(); i < Parser_->Argc_; ++i) {
            v.push_back(Parser_->Argv_[i]);
        }
        return v;
    }

    size_t TOptsParseResult::GetFreeArgCount() const {
        return Parser_->Argc_ - GetFreeArgsPos();
    }

    void FindUserTypos(const TString& arg, const TOpts* options) {
        if (arg.size() < 4 || !arg.StartsWith("-")) {
            return;
        }

        for (auto opt: options->Opts_) {
            for (auto name: opt->GetLongNames()) {
                if ("-" + name == arg) {
                    throw TUsageException() << "did you mean `-" << arg << "` (with two dashes)?";
                }
            }
        }
    }

    void TOptsParseResult::Init(const TOpts* options, int argc, const char** argv) {
        try {
            Parser_.Reset(new TOptsParser(options, argc, argv));
            while (Parser_->Next()) {
                TOptParseResult& r = OptParseResult();
                r.AddValue(Parser_->CurValOrOpt().data());
            }

            Y_ENSURE(options);
            const auto freeArgs = GetFreeArgs();
            for (size_t i = 0; i < freeArgs.size(); ++i) {
                if (i >= options->ArgBindings_.size()) {
                    break;
                }

                options->ArgBindings_[i](freeArgs[i]);
            }

            if (options->CheckUserTypos_) {
                for (auto arg: TVector<TString>(argv, std::next(argv, argc))) {
                    FindUserTypos(arg, options);
                }
            }
        } catch (...) {
            HandleError();
        }
    }

    void TOptsParseResult::HandleError() const {
        Cerr << CurrentExceptionMessage() << Endl;
        if (Parser_.Get()) { // parser initializing can fail (and we get here, see Init)
            if (Parser_->Opts_->FindLongOption("help") != nullptr) {
                Cerr << "Try '" << Parser_->ProgramName_ << " --help' for more information." << Endl;
            } else {
                PrintUsage();
            }
        }
        exit(1);
    }

    void TOptsParseResultException::HandleError() const {
        throw;
    }

}
