#include "completer_command.h"
#include "last_getopt_opts.h"
#include "wrap.h"
#include "last_getopt_parser.h"

#include <library/cpp/colorizer/colors.h>

#include <util/stream/format.h>
#include <util/charset/utf8.h>

#include <stdlib.h>

namespace NLastGetoptPrivate {
    TString& VersionString() {
        static TString data;
        return data;
    }
    TString& ShortVersionString() {
        static TString data;
        return data;
    }
}

namespace NLastGetopt {
    static const TStringBuf SPad = "  ";

    void PrintVersionAndExit(const TOptsParser*) {
        Cout << (NLastGetoptPrivate::VersionString() ? NLastGetoptPrivate::VersionString() : "program version: not linked with library/cpp/getopt") << Endl;
        exit(NLastGetoptPrivate::VersionString().empty());
    }

    void PrintShortVersionAndExit(const TString& appName) {
        Cout << appName << " version " << (NLastGetoptPrivate::ShortVersionString() ? NLastGetoptPrivate::ShortVersionString() : "not linked with library/cpp/getopt") << Endl;
        exit(NLastGetoptPrivate::ShortVersionString().empty());
    }

    // Like TString::Quote(), but does not quote digits-only string
    static TString QuoteForHelp(const TString& str) {
        if (str.empty())
            return str.Quote();
        for (size_t i = 0; i < str.size(); ++i) {
            if (!isdigit(str[i]))
                return str.Quote();
        }
        return str;
    }

    namespace NPrivate {
        TString OptToString(char c) {
            TStringStream ss;
            ss << "-" << c;
            return ss.Str();
        }

        TString OptToString(const TString& longOption) {
            TStringStream ss;
            ss << "--" << longOption;
            return ss.Str();
        }

        TString OptToString(const TOpt* opt) {
            return opt->ToShortString();
        }
    }

    TOpts::TOpts(const TStringBuf& optstring)
        : ArgPermutation_(DEFAULT_ARG_PERMUTATION)
        , AllowSingleDashForLong_(false)
        , AllowPlusForLong_(false)
        , AllowUnknownCharOptions_(false)
        , AllowUnknownLongOptions_(false)
        , FreeArgsMin_(0)
        , FreeArgsMax_(UNLIMITED_ARGS)
    {
        if (!optstring.empty()) {
            AddCharOptions(optstring);
        }
        AddVersionOption(0);
    }

    void TOpts::AddCharOptions(const TStringBuf& optstring) {
        size_t p = 0;
        if (optstring[p] == '+') {
            ArgPermutation_ = REQUIRE_ORDER;
            ++p;
        } else if (optstring[p] == '-') {
            ArgPermutation_ = RETURN_IN_ORDER;
            ++p;
        }

        while (p < optstring.size()) {
            char c = optstring[p];
            p++;
            EHasArg ha = NO_ARGUMENT;
            if (p < optstring.size() && optstring[p] == ':') {
                ha = REQUIRED_ARGUMENT;
                p++;
            }
            if (p < optstring.size() && optstring[p] == ':') {
                ha = OPTIONAL_ARGUMENT;
                p++;
            }
            AddCharOption(c, ha);
        }
    }

    const TOpt* TOpts::FindLongOption(const TStringBuf& name) const {
        for (const auto& Opt : Opts_) {
            const TOpt* opt = Opt.Get();
            if (IsIn(opt->GetLongNames(), name))
                return opt;
        }
        return nullptr;
    }

    TOpt* TOpts::FindLongOption(const TStringBuf& name) {
        for (auto& Opt : Opts_) {
            TOpt* opt = Opt.Get();
            if (IsIn(opt->GetLongNames(), name))
                return opt;
        }
        return nullptr;
    }

    const TOpt* TOpts::FindCharOption(char c) const {
        for (const auto& Opt : Opts_) {
            const TOpt* opt = Opt.Get();
            if (IsIn(opt->GetShortNames(), c))
                return opt;
        }
        return nullptr;
    }

    TOpt* TOpts::FindCharOption(char c) {
        for (auto& Opt : Opts_) {
            TOpt* opt = Opt.Get();
            if (IsIn(opt->GetShortNames(), c))
                return opt;
        }
        return nullptr;
    }

    const TOpt& TOpts::GetCharOption(char c) const {
        const TOpt* option = FindCharOption(c);
        if (!option)
            ythrow TException() << "unknown char option '" << c << "'";
        return *option;
    }

    TOpt& TOpts::GetCharOption(char c) {
        TOpt* option = FindCharOption(c);
        if (!option)
            ythrow TException() << "unknown char option '" << c << "'";
        return *option;
    }

    const TOpt& TOpts::GetLongOption(const TStringBuf& name) const {
        const TOpt* option = FindLongOption(name);
        if (!option)
            ythrow TException() << "unknown option " << name;
        return *option;
    }

    TOpt& TOpts::GetLongOption(const TStringBuf& name) {
        TOpt* option = FindLongOption(name);
        if (!option)
            ythrow TException() << "unknown option " << name;
        return *option;
    }

    bool TOpts::HasAnyShortOption() const {
        for (const auto& Opt : Opts_) {
            const TOpt* opt = Opt.Get();
            if (!opt->GetShortNames().empty())
                return true;
        }
        return false;
    }

    bool TOpts::HasAnyLongOption() const {
        for (const auto& Opt : Opts_) {
            TOpt* opt = Opt.Get();
            if (!opt->GetLongNames().empty())
                return true;
        }
        return false;
    }

    void TOpts::Validate() const {
        for (TOptsVector::const_iterator i = Opts_.begin(); i != Opts_.end(); ++i) {
            TOpt* opt = i->Get();
            const TOpt::TShortNames& shortNames = opt->GetShortNames();
            for (auto c : shortNames) {
                for (TOptsVector::const_iterator j = i + 1; j != Opts_.end(); ++j) {
                    TOpt* nextOpt = j->Get();
                    if (nextOpt->CharIs(c))
                        ythrow TConfException() << "option "
                                                << NPrivate::OptToString(c)
                                                << " is defined more than once";
                }
            }
            const TOpt::TLongNames& longNames = opt->GetLongNames();
            for (const auto& longName : longNames) {
                for (TOptsVector::const_iterator j = i + 1; j != Opts_.end(); ++j) {
                    TOpt* nextOpt = j->Get();
                    if (nextOpt->NameIs(longName))
                        ythrow TConfException() << "option "
                                                << NPrivate::OptToString(longName)
                                                << " is defined more than once";
                }
            }
        }
        if (FreeArgsMax_ < FreeArgsMin_) {
            ythrow TConfException() << "FreeArgsMax must be >= FreeArgsMin";
        }
        if (!FreeArgSpecs_.empty() && FreeArgSpecs_.rbegin()->first >= FreeArgsMax_) {
            ythrow TConfException() << "Described args count is greater than FreeArgsMax. Either increase FreeArgsMax or remove unreachable descriptions";
        }
    }

    TOpt& TOpts::AddOption(const TOpt& option) {
        if (option.GetShortNames().empty() && option.GetLongNames().empty())
            ythrow TConfException() << "bad option: no chars, no long names";
        Opts_.push_back(new TOpt(option));
        return *Opts_.back();
    }

    TOpt& TOpts::AddCompletionOption(TString command, TString longName) {
        if (TOpt* o = FindLongOption(longName)) {
            return *o;
        }

        return AddOption(MakeCompletionOpt(this, std::move(command), std::move(longName)));
    }

    namespace {
        auto MutuallyExclusiveHandler(const TOpt* cur, const TOpt* other) {
            return [cur, other](const TOptsParser* p) {
                if (p->Seen(other)) {
                    throw TUsageException()
                        << "option " << cur->ToShortString()
                        << " can't appear together with option " << other->ToShortString();
                }
            };
        }
    }

    void TOpts::MutuallyExclusiveOpt(TOpt& opt1, TOpt& opt2) {
        opt1.Handler1(MutuallyExclusiveHandler(&opt1, &opt2))
            .IfPresentDisableCompletionFor(opt2);
        opt2.Handler1(MutuallyExclusiveHandler(&opt2, &opt1))
            .IfPresentDisableCompletionFor(opt1);
    }

    size_t TOpts::IndexOf(const TOpt* opt) const {
        TOptsVector::const_iterator it = std::find(Opts_.begin(), Opts_.end(), opt);
        if (it == Opts_.end())
            ythrow TException() << "unknown option";
        return it - Opts_.begin();
    }

    TStringBuf TOpts::GetFreeArgTitle(size_t pos) const {
        if (FreeArgSpecs_.contains(pos)) {
            return FreeArgSpecs_.at(pos).GetTitle(DefaultFreeArgTitle_);
        }
        return DefaultFreeArgTitle_;
    }

    void TOpts::SetFreeArgTitle(size_t pos, const TString& title, const TString& help, bool optional) {
        FreeArgSpecs_[pos] = TFreeArgSpec(title, help, optional);
    }

    TFreeArgSpec& TOpts::GetFreeArgSpec(size_t pos) {
        return FreeArgSpecs_[pos];
    }

    static TString FormatOption(const TOpt* option, const NColorizer::TColors& colors) {
        TStringStream result;
        const TOpt::TShortNames& shorts = option->GetShortNames();
        const TOpt::TLongNames& longs = option->GetLongNames();

        const size_t nopts = shorts.size() + longs.size();
        const bool multiple = 1 < nopts;
        if (multiple)
            result << '{';
        for (size_t i = 0; i < nopts; ++i) {
            if (multiple && 0 != i)
                result << '|';

            if (i < shorts.size()) // short
                result << colors.GreenColor() << '-' << shorts[i] << colors.OldColor();
            else
                result << colors.GreenColor() << "--" << longs[i - shorts.size()] << colors.OldColor();
        }
        if (multiple)
            result << '}';

        static const TString metavarDef("VAL");
        const TString& title = option->GetArgTitle();
        const TString& metavar = title.empty() ? metavarDef : title;

        if (option->GetHasArg() == OPTIONAL_ARGUMENT) {
            if (option->IsEqParseOnly()) {
                result << "[=";
            } else {
                result << " [";
            }
            result << metavar;
            if (option->HasOptionalValue())
                result << ':' << option->GetOptionalValue();
            result << ']';
        } else if (option->GetHasArg() == REQUIRED_ARGUMENT) {
            if (option->IsEqParseOnly()) {
                result << "=";
            } else {
                result << " ";
            }
            result << metavar;
        } else
            Y_ASSERT(option->GetHasArg() == NO_ARGUMENT);

        return result.Str();
    }

    void TOpts::PrintCmdLine(const TStringBuf& program, IOutputStream& os, const NColorizer::TColors& colors) const {
        os << colors.BoldColor() << "Usage" << colors.OldColor() << ": ";
        if (CustomUsage) {
            os << CustomUsage;
        } else {
            os << program << " ";
        }
        if (CustomCmdLineDescr) {
            os << CustomCmdLineDescr << Endl;
            return;
        }
        os << "[OPTIONS]";

        ui32 numDescribedFlags = FreeArgSpecs_.empty() ? 0 : FreeArgSpecs_.rbegin()->first + 1;
        ui32 numArgsToShow = Max(FreeArgsMin_, FreeArgsMax_ == UNLIMITED_ARGS ? numDescribedFlags : FreeArgsMax_);

        for (ui32 i = 0, nonOptionalFlagsPrinted = 0; i < numArgsToShow; ++i) {
            bool isOptional = nonOptionalFlagsPrinted >= FreeArgsMin_ || FreeArgSpecs_.Value(i, TFreeArgSpec()).Optional_;

            nonOptionalFlagsPrinted += !isOptional;

            os << " ";

            if (isOptional)
                os << "[";

            os << GetFreeArgTitle(i);

            if (isOptional)
                os << "]";
        }

        if (FreeArgsMax_ == UNLIMITED_ARGS) {
            os << " [" << TrailingArgSpec_.GetTitle(DefaultFreeArgTitle_) << "]...";
        }

        os << Endl;
    }

    void TOpts::PrintUsage(const TStringBuf& program, IOutputStream& osIn, const NColorizer::TColors& colors) const {
        TStringStream os;

        if (!Title.empty())
            os << Title << "\n\n";

        PrintCmdLine(program, os, colors);

        TVector<TString> leftColumn(Opts_.size());
        TVector<size_t> leftColumnSizes(leftColumn.size());
        const size_t kMaxLeftWidth = 25;
        size_t leftWidth = 0;
        size_t requiredOptionsCount = 0;
        NColorizer::TColors disabledColors(false);

        for (size_t i = 0; i < Opts_.size(); i++) {
            const TOpt* opt = Opts_[i].Get();
            if (opt->IsHidden())
                continue;
            leftColumn[i] = FormatOption(opt, colors);
            size_t leftColumnSize = leftColumn[i].size();
            if (colors.IsTTY()) {
                leftColumnSize -= NColorizer::TotalAnsiEscapeCodeLen(leftColumn[i]);
            }
            leftColumnSizes[i] = leftColumnSize;
            if (leftColumnSize <= kMaxLeftWidth) {
                leftWidth = Max(leftWidth, leftColumnSize);
            }
            if (opt->IsRequired())
                requiredOptionsCount++;
        }

        const TString leftPadding(leftWidth, ' ');

        for (size_t sectionId = 0; sectionId <= 1; sectionId++) {
            bool requiredOptionsSection = (sectionId == 0);

            if (requiredOptionsSection) {
                if (requiredOptionsCount == 0)
                    continue;
                os << Endl << colors.BoldColor() << "Required parameters" << colors.OldColor() << ":" << Endl;
            } else {
                if (requiredOptionsCount == Opts_.size())
                    continue;
                if (requiredOptionsCount == 0)
                    os << Endl << colors.BoldColor() << "Options" << colors.OldColor() << ":" << Endl;
                else
                    os << Endl << colors.BoldColor() << "Optional parameters" << colors.OldColor() << ":" << Endl; // optional options would be a tautology
            }

            for (size_t i = 0; i < Opts_.size(); i++) {
                const TOpt* opt = Opts_[i].Get();

                if (opt->IsHidden())
                    continue;
                if (opt->IsRequired() != requiredOptionsSection)
                    continue;

                if (leftColumnSizes[i] > leftWidth && !opt->GetHelp().empty()) {
                    os << SPad << leftColumn[i] << Endl << SPad << leftPadding << ' ';
                } else {
                    os << SPad << leftColumn[i] << ' ';
                    if (leftColumnSizes[i] < leftWidth)
                        os << TStringBuf(leftPadding.data(), leftWidth - leftColumnSizes[i]);
                }

                TStringBuf help = opt->GetHelp();
                while (help && isspace(help.back())) {
                    help.Chop(1);
                }
                size_t lastLineLength = 0;
                bool helpHasParagraphs = false;
                if (help) {
                    os << Wrap(Wrap_, help, SPad + leftPadding + " ", &lastLineLength, &helpHasParagraphs);
                }

                auto choicesHelp = opt->GetChoicesHelp();
                if (!choicesHelp.empty()) {
                    if (help) {
                        os << Endl << SPad << leftPadding << " ";
                    }
                    os << "(values: " << choicesHelp << ")";
                }

                if (opt->HasDefaultValue()) {
                    auto quotedDef = QuoteForHelp(opt->GetDefaultValue());
                    if (helpHasParagraphs) {
                        os << Endl << Endl << SPad << leftPadding << " ";
                        os << "Default: " << colors.CyanColor() << quotedDef << colors.OldColor() << ".";
                    } else if (help.EndsWith('.')) {
                        os << Endl << SPad << leftPadding << " ";
                        os << "Default: " << colors.CyanColor() << quotedDef << colors.OldColor() << ".";
                    } else if (help) {
                        if (SPad.size() + leftWidth + 1 + lastLineLength + 12 + quotedDef.size() > Wrap_) {
                            os << Endl << SPad << leftPadding << " ";
                        } else {
                            os << " ";
                        }
                        os << "(default: " << colors.CyanColor() << quotedDef << colors.OldColor() << ")";
                    } else {
                        os << "default: " << colors.CyanColor() << quotedDef << colors.OldColor();
                    }
                }

                os << Endl;

                if (helpHasParagraphs) {
                    os << Endl;
                }
            }
        }

        PrintFreeArgsDesc(os, colors);

        for (auto& [heading, text] : Sections) {
            os << Endl << colors.BoldColor() << heading << colors.OldColor() << ":" << Endl;

            os << SPad << Wrap(Wrap_, text, SPad) << Endl;
        }

        osIn << os.Str();
    }

    void TOpts::PrintUsage(const TStringBuf& program, IOutputStream& os) const {
        PrintUsage(program, os, NColorizer::AutoColors(os));
    }

    void TOpts::PrintFreeArgsDesc(IOutputStream& os, const NColorizer::TColors& colors) const {
        if (0 == FreeArgsMax_)
            return;

        size_t leftFreeWidth = 0;
        for (size_t i = 0; i < FreeArgSpecs_.size(); ++i) {
            leftFreeWidth = Max(leftFreeWidth, GetFreeArgTitle(i).size());
        }

        if (!TrailingArgSpec_.IsDefault()) {
            leftFreeWidth = Max(leftFreeWidth, TrailingArgSpec_.GetTitle(DefaultFreeArgTitle_).size());
        }

        leftFreeWidth = Min(leftFreeWidth, size_t(30));
        os << Endl << colors.BoldColor() << "Free args" << colors.OldColor() << ":";

        os << " min: " << colors.GreenColor() << FreeArgsMin_ << colors.OldColor() << ",";
        os << " max: " << colors.GreenColor();
        if (FreeArgsMax_ != UNLIMITED_ARGS) {
            os << FreeArgsMax_;
        } else {
            os << "unlimited";
        }
        os << colors.OldColor() << Endl;

        const size_t limit = FreeArgSpecs_.empty() ? 0 : FreeArgSpecs_.rbegin()->first;
        for (size_t i = 0; i <= limit; ++i) {
            if (!FreeArgSpecs_.contains(i)) {
                continue;
            }

            if (auto help = FreeArgSpecs_.at(i).GetHelp()) {
                auto title = GetFreeArgTitle(i);
                os << SPad << colors.GreenColor() << RightPad(title, leftFreeWidth, ' ') << colors.OldColor()
                   << SPad << help << Endl;
            }
        }

        if (FreeArgsMax_ == UNLIMITED_ARGS) {
            auto title = TrailingArgSpec_.GetTitle(DefaultFreeArgTitle_);
            if (auto help = TrailingArgSpec_.GetHelp()) {
                os << SPad << colors.GreenColor() << RightPad(title, leftFreeWidth, ' ') << colors.OldColor()
                   << SPad << help << Endl;
            }
        }
    }
}
