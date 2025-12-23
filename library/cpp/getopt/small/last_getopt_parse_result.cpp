#include "last_getopt_parse_result.h"

namespace NLastGetopt {
    const TOptParseResult* TOptsParseResult::FindParseResult(const TdVec& vec, const TOpt* opt) {
        for (const auto& r : vec) {
            if (r.OptPtr() == opt)
                return &r;
        }
        return nullptr;
    }

    void TOptsParseResult::BuildTaggedFreeArgs(const TOpts* options) {
        TaggedFreeArgs_.clear();

        if (!Parser_) {
            return;
        }

        const size_t freeArgsPos = GetFreeArgsPos();
        for (size_t argPos = freeArgsPos; argPos < Parser_->Argc_; ++argPos) {
            size_t index = argPos - freeArgsPos;

            TString value = Parser_->Argv_[argPos];
            ui32 tag = 0;

            if (options) {
                const TFreeArgSpec* spec = nullptr;
                auto it = options->FreeArgSpecs_.find(index);
                if (it != options->FreeArgSpecs_.end()) {
                    spec = &it->second;
                } else if (options->FreeArgsMax_ == TOpts::UNLIMITED_ARGS) {
                    ui32 trailingArgsIndex = options->GetTrailingArgsIndex();
                    if (index >= trailingArgsIndex) {
                        spec = &options->TrailingArgSpec_;
                    }
                }

                if (spec) {
                    tag = spec->GetTag(value);
                }
            }

            TaggedFreeArgs_.push_back(TTaggedArg {
                .Value = value,
                .Tag = tag
            });
        }
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

    void TOptsParseResult::SetProgramSubcommandPath(const TVector<TString>& parts) {
        ProgramSubcommandPath_ = parts;
    }

    const TVector<TString>& TOptsParseResult::GetProgramSubcommandPath() const {
        return ProgramSubcommandPath_;
    }

    void TOptsParseResult::PrintUsage(IOutputStream& os) const {
        Parser_->Opts_->PrintUsage(Parser_->ProgramName_, os);
    }

    size_t TOptsParseResult::GetFreeArgsPos() const {
        return Parser_->Pos_;
    }

    TVector<TString> TOptsParseResult::GetFreeArgs() const {
        TVector<TString> args;
        args.reserve(TaggedFreeArgs_.size());
        for (const auto& arg : TaggedFreeArgs_) {
            args.push_back(arg.Value);
        }
        return args;
    }

    size_t TOptsParseResult::GetFreeArgCount() const {
        return TaggedFreeArgs_.size();
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
            BuildTaggedFreeArgs(options);
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
