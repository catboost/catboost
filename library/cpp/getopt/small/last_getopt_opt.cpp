#include "last_getopt_opt.h"

#include <util/stream/format.h>
#include <util/string/escape.h>
#include <util/generic/ylimits.h>
#include <util/generic/utility.h>
#include <util/generic/algorithm.h>
#include <ctype.h>

namespace NLastGetopt {
    static const TStringBuf ExcludedShortNameChars = "= -\t\n";
    static const TStringBuf ExcludedLongNameChars = "= \t\n";

    bool TOpt::NameIs(const TString& name) const {
        for (const auto& next : LongNames_) {
            if (next == name)
                return true;
        }
        return false;
    }

    bool TOpt::CharIs(char c) const {
        for (auto next : Chars_) {
            if (next == c)
                return true;
        }
        return false;
    }

    char TOpt::GetChar() const {
        if (Chars_.empty())
            ythrow TConfException() << "no char for option " << this->ToShortString();
        return Chars_.at(0);
    }

    char TOpt::GetCharOr0() const {
        if (Chars_.empty())
            return 0;
        return GetChar();
    }

    TString TOpt::GetName() const {
        if (LongNames_.empty())
            ythrow TConfException() << "no name for option " << this->ToShortString();
        return LongNames_.at(0);
    }

    bool TOpt::IsAllowedShortName(unsigned char c) {
        return isprint(c) && TStringBuf::npos == ExcludedShortNameChars.find(c);
    }

    TOpt& TOpt::AddShortName(unsigned char c) {
        if (!IsAllowedShortName(c))
            throw TUsageException() << "option char '" << c << "' is not allowed";
        Chars_.push_back(c);
        return *this;
    }

    bool TOpt::IsAllowedLongName(const TString& name, unsigned char* out) {
        for (size_t i = 0; i != name.size(); ++i) {
            const unsigned char c = name[i];
            if (!isprint(c) || TStringBuf::npos != ExcludedLongNameChars.find(c)) {
                if (nullptr != out)
                    *out = c;
                return false;
            }
        }
        return true;
    }

    TOpt& TOpt::AddLongName(const TString& name) {
        unsigned char c = 0;
        if (!IsAllowedLongName(name, &c))
            throw TUsageException() << "option char '" << c
                                     << "' in long '" << name << "' is not allowed";
        LongNames_.push_back(name);
        return *this;
    }

    namespace NPrivate {
        TString OptToString(char c);

        TString OptToString(const TString& longOption);
    }

    TString TOpt::ToShortString() const {
        if (!LongNames_.empty())
            return NPrivate::OptToString(LongNames_.front());
        if (!Chars_.empty())
            return NPrivate::OptToString(Chars_.front());
        return "?";
    }

    void TOpt::FireHandlers(const TOptsParser* parser) const {
        for (const auto& handler : Handlers_) {
            handler->HandleOpt(parser);
        }
    }

    TOpt& TOpt::IfPresentDisableCompletionFor(const TOpt& opt) {
        if (opt.GetLongNames()) {
            IfPresentDisableCompletionFor(opt.GetName());
        } else {
            IfPresentDisableCompletionFor(opt.GetChar());
        }
        return *this;
    }
}
