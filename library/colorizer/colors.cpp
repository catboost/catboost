#include "colors.h"

#include <util/stream/output.h>
#include <util/string/builder.h>
#include <util/generic/singleton.h>
#include <util/system/env.h>

#include <cstdlib>

#if defined(_unix_)
    #include <unistd.h>
#endif

using namespace NColorizer;

TColors::TColors(FILE* f)
    : IsTTY_(true)
{
    InitNames();
    SetIsTTY(CalcIsTTY(f));
}

TColors::TColors(bool ontty)
    : IsTTY_(true)
{
    InitNames();
    SetIsTTY(ontty);
}

TStringBuf TColors::OldColor() const {
    return Color(39);
}

TStringBuf TColors::BoldColor() const {
    return Color(1);
}

TStringBuf TColors::BlackColor() const {
    return Color(30);
}

TStringBuf TColors::BlueColor() const {
    return Color(34);
}

TStringBuf TColors::GreenColor() const {
    return Color(32);
}

TStringBuf TColors::CyanColor() const {
    return Color(36);
}

TStringBuf TColors::RedColor() const {
    return Color(31);
}

TStringBuf TColors::PurpleColor() const {
    return Color(35);
}

TStringBuf TColors::BrownColor() const {
    return Color(33);
}

TStringBuf TColors::LightGrayColor() const {
    return Color(37);
}

TStringBuf TColors::DarkGrayColor() const {
    return Color(30, 1);
}

TStringBuf TColors::LightBlueColor() const {
    return Color(34, 1);
}

TStringBuf TColors::LightGreenColor() const {
    return Color(32, 1);
}

TStringBuf TColors::LightCyanColor() const {
    return Color(36, 1);
}

TStringBuf TColors::LightRedColor() const {
    return Color(31, 1);
}

TStringBuf TColors::LightPurpleColor() const {
    return Color(35, 1);
}

TStringBuf TColors::YellowColor() const {
    return Color(33, 1);
}

TStringBuf TColors::WhiteColor() const {
    return Color(37, 1);
}

TStringBuf TColors::Color(const TStringBuf& name) const {
    if (IsTTY()) {
        const TStringBuf* ret = ByName.FindPtr(name);

        if (ret) {
            return *ret;
        }

        ythrow yexception() << "unknown color name " << name << " (see TColors::InitNames for the list of names)";
    }

    return TStringBuf();
}

TStringBuf TColors::ColorImpl(int val, int bold) const {
    size_t idx = val * 2 + bold;
    {
        TReadGuard guard(Mutex);
        yhash<int, TString>::iterator i = ColorBufs.find(idx);
        if (!i.IsEnd())
            return i->second;
    }
    TWriteGuard guard(Mutex);
    TString& ret = ColorBufs[idx];

    if (ret.empty()) {
        if (val == 0 || val == 1) {
            ret = TStringBuilder() << STRINGBUF("\033[") << val << STRINGBUF("m");
        } else {
            ret = TStringBuilder() << STRINGBUF("\033[") << (bold ? 1 : 22) << STRINGBUF(";") << val << STRINGBUF("m");
        }
    }

    return ret;
}

bool TColors::CalcIsTTY(FILE* file) {
    if (GetEnv("ENFORCE_TTY")) {
        return true;
    }

#if defined(_unix_)
    return isatty(fileno(file));
#endif
    Y_UNUSED(file);
    return false;
}

void TColors::InitNames() {
    ByName["black"] = BlackColor();
    ByName["blue"] = BlueColor();
    ByName["green"] = GreenColor();
    ByName["cyan"] = CyanColor();
    ByName["red"] = RedColor();
    ByName["purple"] = PurpleColor();
    ByName["brown"] = BrownColor();
    ByName["light-gray"] = LightGrayColor();
    ByName["dark-gray"] = LightGrayColor();
    ByName["light-blue"] = LightBlueColor();
    ByName["light-green"] = LightGreenColor();
    ByName["light-cyan"] = LightCyanColor();
    ByName["light-red"] = LightRedColor();
    ByName["light-purple"] = LightPurpleColor();
    ByName["yellow"] = YellowColor();
    ByName["white"] = WhiteColor();
}

namespace {
    using TStdErrColors = TColors;

    class TStdOutColors: public TColors {
    public:
        TStdOutColors()
            : TColors(stdout)
        {
        }
    };

    class TDisabledColors: public TColors {
    public:
        TDisabledColors()
            : TColors(false)
        {
        }
    };
} // anonymous namespace

TColors& NColorizer::AutoColors(IOutputStream& os) {
    if (&os == &Cerr) {
        return StdErr();
    }
    if (&os == &Cout) {
        return StdOut();
    }
    return *Singleton<TDisabledColors>();
}

TColors& NColorizer::StdErr() {
    return *Singleton<TStdErrColors>();
}

TColors& NColorizer::StdOut() {
    return *Singleton<TStdOutColors>();
}
