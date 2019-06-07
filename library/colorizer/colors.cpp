#include "colors.h"

#include <util/stream/output.h>
#include <util/generic/singleton.h>
#include <util/system/env.h>

#include <cstdlib>

#if defined(_unix_)
#include <unistd.h>
#endif

using namespace NColorizer;

namespace {
    constexpr TStringBuf ToStringBufC(NColorizer::EAnsiCode x) {
        switch(x) {
            case RESET:
                return AsStringBuf("\033[0m");

            case ST_LIGHT:
                return AsStringBuf("\033[1m");
            case ST_DARK:
                return AsStringBuf("\033[2m");
            case ST_NORMAL:
                return AsStringBuf("\033[22m");

            case ITALIC_ON:
                return AsStringBuf("\033[3m");
            case ITALIC_OFF:
                return AsStringBuf("\033[23m");
            case UNDERLINE_ON:
                return AsStringBuf("\033[4m");
            case UNDERLINE_OFF:
                return AsStringBuf("\033[24m");

            case FG_DEFAULT:
                return AsStringBuf("\033[39m");
            case FG_BLACK:
                return AsStringBuf("\033[30m");
            case FG_RED:
                return AsStringBuf("\033[31m");
            case FG_GREEN:
                return AsStringBuf("\033[32m");
            case FG_YELLOW:
                return AsStringBuf("\033[33m");
            case FG_BLUE:
                return AsStringBuf("\033[34m");
            case FG_MAGENTA:
                return AsStringBuf("\033[35m");
            case FG_CYAN:
                return AsStringBuf("\033[36m");
            case FG_WHITE:
                return AsStringBuf("\033[37m");

            case BG_DEFAULT:
                return AsStringBuf("\033[49m");
            case BG_BLACK:
                return AsStringBuf("\033[40m");
            case BG_RED:
                return AsStringBuf("\033[41m");
            case BG_GREEN:
                return AsStringBuf("\033[42m");
            case BG_YELLOW:
                return AsStringBuf("\033[43m");
            case BG_BLUE:
                return AsStringBuf("\033[44m");
            case BG_MAGENTA:
                return AsStringBuf("\033[45m");
            case BG_CYAN:
                return AsStringBuf("\033[46m");
            case BG_WHITE:
                return AsStringBuf("\033[47m");

            // Note: the following codes are split into two escabe sequences because of how ya.make handles them.

            case DEFAULT:
                return AsStringBuf("\033[0m\033[0;39m");
            case BLACK:
                return AsStringBuf("\033[0m\033[0;30m");
            case RED:
                return AsStringBuf("\033[0m\033[0;31m");
            case GREEN:
                return AsStringBuf("\033[0m\033[0;32m");
            case YELLOW:
                return AsStringBuf("\033[0m\033[0;33m");
            case BLUE:
                return AsStringBuf("\033[0m\033[0;34m");
            case MAGENTA:
                return AsStringBuf("\033[0m\033[0;35m");
            case CYAN:
                return AsStringBuf("\033[0m\033[0;36m");
            case WHITE:
                return AsStringBuf("\033[0m\033[0;37m");

            case LIGHT_DEFAULT:
                return AsStringBuf("\033[0m\033[1;39m");
            case LIGHT_BLACK:
                return AsStringBuf("\033[0m\033[1;30m");
            case LIGHT_RED:
                return AsStringBuf("\033[0m\033[1;31m");
            case LIGHT_GREEN:
                return AsStringBuf("\033[0m\033[1;32m");
            case LIGHT_YELLOW:
                return AsStringBuf("\033[0m\033[1;33m");
            case LIGHT_BLUE:
                return AsStringBuf("\033[0m\033[1;34m");
            case LIGHT_MAGENTA:
                return AsStringBuf("\033[0m\033[1;35m");
            case LIGHT_CYAN:
                return AsStringBuf("\033[0m\033[1;36m");
            case LIGHT_WHITE:
                return AsStringBuf("\033[0m\033[1;37m");

            case DARK_DEFAULT:
                return AsStringBuf("\033[0m\033[2;39m");
            case DARK_BLACK:
                return AsStringBuf("\033[0m\033[2;30m");
            case DARK_RED:
                return AsStringBuf("\033[0m\033[2;31m");
            case DARK_GREEN:
                return AsStringBuf("\033[0m\033[2;32m");
            case DARK_YELLOW:
                return AsStringBuf("\033[0m\033[2;33m");
            case DARK_BLUE:
                return AsStringBuf("\033[0m\033[2;34m");
            case DARK_MAGENTA:
                return AsStringBuf("\033[0m\033[2;35m");
            case DARK_CYAN:
                return AsStringBuf("\033[0m\033[2;36m");
            case DARK_WHITE:
                return AsStringBuf("\033[0m\033[2;37m");

            case INVALID:
            default:
                return AsStringBuf("");
        }
    }
}

TStringBuf ToStringBuf(NColorizer::EAnsiCode x) {
    return ToStringBufC(x);
}

TString ToString(NColorizer::EAnsiCode x) {
    return TString(ToStringBufC(x));
}

template<>
void Out<NColorizer::EAnsiCode>(IOutputStream& os, TTypeTraits<NColorizer::EAnsiCode>::TFuncParam x) {
    if (AutoColors(os).IsTTY()) {
        os << ToStringBufC(x);
    }
}

bool TColors::CalcIsTTY(FILE* file) {
    if (GetEnv("ENFORCE_TTY")) {
        return true;
    }

#if defined(_unix_)
    return isatty(fileno(file));
#else
    Y_UNUSED(file);
    return false;
#endif
}

TColors::TColors(FILE* f)
    : IsTTY_(true)
{
    SetIsTTY(CalcIsTTY(f));
}

TColors::TColors(bool ontty)
    : IsTTY_(true)
{
    SetIsTTY(ontty);
}

TStringBuf TColors::Reset() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::RESET) : ToStringBufC(EAnsiCode::INVALID);
}

TStringBuf TColors::StyleLight() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::ST_LIGHT) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::StyleDark() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::ST_DARK) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::StyleNormal() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::ST_NORMAL) : ToStringBufC(EAnsiCode::INVALID);
}

TStringBuf TColors::ItalicOn() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::ITALIC_ON) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::ItalicOff() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::ITALIC_OFF) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::UnderlineOn() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::UNDERLINE_ON) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::UnderlineOff() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::UNDERLINE_OFF) : ToStringBufC(EAnsiCode::INVALID);
}

TStringBuf TColors::ForeDefault() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::FG_DEFAULT) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::ForeBlack() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::FG_BLACK) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::ForeRed() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::FG_RED) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::ForeGreen() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::FG_GREEN) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::ForeYellow() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::FG_YELLOW) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::ForeBlue() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::FG_BLUE) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::ForeMagenta() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::FG_MAGENTA) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::ForeCyan() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::FG_CYAN) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::ForeWhite() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::FG_WHITE) : ToStringBufC(EAnsiCode::INVALID);
}

TStringBuf TColors::BackDefault() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::BG_DEFAULT) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::BackBlack() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::BG_BLACK) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::BackRed() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::BG_RED) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::BackGreen() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::BG_GREEN) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::BackYellow() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::BG_YELLOW) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::BackBlue() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::BG_BLUE) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::BackMagenta() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::BG_MAGENTA) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::BackCyan() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::BG_CYAN) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::BackWhite() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::BG_WHITE) : ToStringBufC(EAnsiCode::INVALID);
}

TStringBuf TColors::Default() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::DEFAULT) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::Black() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::BLACK) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::Red() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::RED) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::Green() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::GREEN) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::Yellow() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::YELLOW) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::Blue() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::BLUE) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::Magenta() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::MAGENTA) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::Cyan() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::CYAN) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::White() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::WHITE) : ToStringBufC(EAnsiCode::INVALID);
}

TStringBuf TColors::LightDefault() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::LIGHT_DEFAULT) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::LightBlack() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::LIGHT_BLACK) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::LightRed() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::LIGHT_RED) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::LightGreen() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::LIGHT_GREEN) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::LightYellow() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::LIGHT_YELLOW) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::LightBlue() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::LIGHT_BLUE) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::LightMagenta() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::LIGHT_MAGENTA) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::LightCyan() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::LIGHT_CYAN) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::LightWhite() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::LIGHT_WHITE) : ToStringBufC(EAnsiCode::INVALID);
}

TStringBuf TColors::DarkDefault() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::DARK_DEFAULT) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::DarkBlack() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::DARK_BLACK) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::DarkRed() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::DARK_RED) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::DarkGreen() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::DARK_GREEN) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::DarkYellow() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::DARK_YELLOW) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::DarkBlue() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::DARK_BLUE) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::DarkMagenta() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::DARK_MAGENTA) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::DarkCyan() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::DARK_CYAN) : ToStringBufC(EAnsiCode::INVALID);
}
TStringBuf TColors::DarkWhite() const noexcept {
    return IsTTY() ? ToStringBufC(EAnsiCode::DARK_WHITE) : ToStringBufC(EAnsiCode::INVALID);
}

TStringBuf TColors::OldColor() const noexcept {
    return IsTTY() ? AsStringBuf("\033[22;39m") : AsStringBuf("");
}

TStringBuf TColors::BoldColor() const noexcept {
    return IsTTY() ? AsStringBuf("\033[1m") : AsStringBuf("");
}

TStringBuf TColors::BlackColor() const noexcept {
    return IsTTY() ? AsStringBuf("\033[22;30m") : AsStringBuf("");
}

TStringBuf TColors::BlueColor() const noexcept {
    return IsTTY() ? AsStringBuf("\033[22;34m") : AsStringBuf("");
}

TStringBuf TColors::GreenColor() const noexcept {
    return IsTTY() ? AsStringBuf("\033[22;32m") : AsStringBuf("");
}

TStringBuf TColors::CyanColor() const noexcept {
    return IsTTY() ? AsStringBuf("\033[22;36m") : AsStringBuf("");
}

TStringBuf TColors::RedColor() const noexcept {
    return IsTTY() ? AsStringBuf("\033[22;31m") : AsStringBuf("");
}

TStringBuf TColors::PurpleColor() const noexcept {
    return IsTTY() ? AsStringBuf("\033[22;35m") : AsStringBuf("");
}

TStringBuf TColors::BrownColor() const noexcept {
    return IsTTY() ? AsStringBuf("\033[22;33m") : AsStringBuf("");
}

TStringBuf TColors::LightGrayColor() const noexcept {
    return IsTTY() ? AsStringBuf("\033[22;37m") : AsStringBuf("");
}

TStringBuf TColors::DarkGrayColor() const noexcept {
    return IsTTY() ? AsStringBuf("\033[1;30m") : AsStringBuf("");
}

TStringBuf TColors::LightBlueColor() const noexcept {
    return IsTTY() ? AsStringBuf("\033[1;34m") : AsStringBuf("");
}

TStringBuf TColors::LightGreenColor() const noexcept {
    return IsTTY() ? AsStringBuf("\033[1;32m") : AsStringBuf("");
}

TStringBuf TColors::LightCyanColor() const noexcept {
    return IsTTY() ? AsStringBuf("\033[1;36m") : AsStringBuf("");
}

TStringBuf TColors::LightRedColor() const noexcept {
    return IsTTY() ? AsStringBuf("\033[1;31m") : AsStringBuf("");
}

TStringBuf TColors::LightPurpleColor() const noexcept {
    return IsTTY() ? AsStringBuf("\033[1;35m") : AsStringBuf("");
}

TStringBuf TColors::YellowColor() const noexcept {
    return IsTTY() ? AsStringBuf("\033[1;33m") : AsStringBuf("");
}

TStringBuf TColors::WhiteColor() const noexcept {
    return IsTTY() ? AsStringBuf("\033[1;37m") : AsStringBuf("");
}


namespace {
    class TStdErrColors: public TColors {
    public:
        TStdErrColors()
            : TColors(stdout)
        {
        }
    };

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

TColors& NColorizer::StdErr() {
    return *Singleton<TStdErrColors>();
}

TColors& NColorizer::StdOut() {
    return *Singleton<TStdOutColors>();
}

TColors& NColorizer::AutoColors(IOutputStream& os) {
    if (&os == &Cerr) {
        return StdErr();
    }
    if (&os == &Cout) {
        return StdOut();
    }
    return *Singleton<TDisabledColors>();
}
