#pragma once

#include "fwd.h"

#include <util/generic/string.h>
#include <util/generic/strbuf.h>

#include <cstdio>

namespace NColorizer {
    /**
     * List of ECMA-48 colors.
     *
     * When printing elements of this enum via `operator<<`, `AutoColors()` (see below) function will be used
     * to produce colors, i.e. nothing will be printed to non-tty streams. When converting elements of this enum
     * via `ToString`, escape code is always returned.
     *
     * Note: as of now (2019-03), `ya make` strips out some escape codes from compiler output.
     * It also inserts `RESET` before each color code. See https://st.yandex-team.ru/DEVTOOLS-5269 for details.
     * For now, do not use `OLD`, `ST_*`, `FG_*` and `BG_*` in tools that run through `ya make`.
     *
     * Note: refrain from using black colors because there's high chance they'll not be visible on some terminals.
     * Default windows and ubuntu color schemes shows them as black letters on black background.
     * Also, white colors are barely visible in default OSX color scheme. Light black is usually fine though.
     */
    enum EAnsiCode: i8 {
        // Note: not using `GENERATE_ENUM_SERIALIZATION` because serialization generator depends on this library.

        /// Does not change anything.
        INVALID,

        /// Reset all styles and colors. Safe to use in `ya make` tools.
        RESET,

        /// Change style, don't change anything else.
        ST_LIGHT,
        ST_DARK,
        ST_NORMAL,

        /// Additional styles.
        ITALIC_ON,
        ITALIC_OFF,
        UNDERLINE_ON,
        UNDERLINE_OFF,

        /// Change foreground color, don't change anything else.
        FG_DEFAULT,
        FG_BLACK,
        FG_RED,
        FG_GREEN,
        FG_YELLOW,
        FG_BLUE,
        FG_MAGENTA,
        FG_CYAN,
        FG_WHITE,

        /// Change background color, don't change anything else.
        BG_DEFAULT,
        BG_BLACK,
        BG_RED,
        BG_GREEN,
        BG_YELLOW,
        BG_BLUE,
        BG_MAGENTA,
        BG_CYAN,
        BG_WHITE,

        /// Reset all styles and colors, then enable a (possibly light or dark) color. Safe to use in `ya make` tools.
        DEFAULT,
        BLACK,
        RED,
        GREEN,
        YELLOW,
        BLUE,
        MAGENTA,
        CYAN,
        WHITE,
        LIGHT_DEFAULT,
        LIGHT_BLACK,
        LIGHT_RED,
        LIGHT_GREEN,
        LIGHT_YELLOW,
        LIGHT_BLUE,
        LIGHT_MAGENTA,
        LIGHT_CYAN,
        LIGHT_WHITE,
        DARK_DEFAULT,
        DARK_BLACK,
        DARK_RED,
        DARK_GREEN,
        DARK_YELLOW,
        DARK_BLUE,
        DARK_MAGENTA,
        DARK_CYAN,
        DARK_WHITE,
    };

    /**
     * Produces escape codes or empty stringbuf depending on settings.
     * All color functions return zero-terminated stringbuf.
     */
    class TColors {
    public:
        static bool CalcIsTTY(FILE* file);

    public:
        explicit TColors(FILE* = stderr);
        explicit TColors(bool ontty);

        TStringBuf Reset() const noexcept;

        TStringBuf StyleLight() const noexcept;
        TStringBuf StyleDark() const noexcept;
        TStringBuf StyleNormal() const noexcept;

        TStringBuf ItalicOn() const noexcept;
        TStringBuf ItalicOff() const noexcept;
        TStringBuf UnderlineOn() const noexcept;
        TStringBuf UnderlineOff() const noexcept;

        TStringBuf ForeDefault() const noexcept;
        TStringBuf ForeBlack() const noexcept;
        TStringBuf ForeRed() const noexcept;
        TStringBuf ForeGreen() const noexcept;
        TStringBuf ForeYellow() const noexcept;
        TStringBuf ForeBlue() const noexcept;
        TStringBuf ForeMagenta() const noexcept;
        TStringBuf ForeCyan() const noexcept;
        TStringBuf ForeWhite() const noexcept;

        TStringBuf BackDefault() const noexcept;
        TStringBuf BackBlack() const noexcept;
        TStringBuf BackRed() const noexcept;
        TStringBuf BackGreen() const noexcept;
        TStringBuf BackYellow() const noexcept;
        TStringBuf BackBlue() const noexcept;
        TStringBuf BackMagenta() const noexcept;
        TStringBuf BackCyan() const noexcept;
        TStringBuf BackWhite() const noexcept;

        TStringBuf Default() const noexcept;
        TStringBuf Black() const noexcept;
        TStringBuf Red() const noexcept;
        TStringBuf Green() const noexcept;
        TStringBuf Yellow() const noexcept;
        TStringBuf Blue() const noexcept;
        TStringBuf Magenta() const noexcept;
        TStringBuf Cyan() const noexcept;
        TStringBuf White() const noexcept;

        TStringBuf LightDefault() const noexcept;
        TStringBuf LightBlack() const noexcept;
        TStringBuf LightRed() const noexcept;
        TStringBuf LightGreen() const noexcept;
        TStringBuf LightYellow() const noexcept;
        TStringBuf LightBlue() const noexcept;
        TStringBuf LightMagenta() const noexcept;
        TStringBuf LightCyan() const noexcept;
        TStringBuf LightWhite() const noexcept;

        TStringBuf DarkDefault() const noexcept;
        TStringBuf DarkBlack() const noexcept;
        TStringBuf DarkRed() const noexcept;
        TStringBuf DarkGreen() const noexcept;
        TStringBuf DarkYellow() const noexcept;
        TStringBuf DarkBlue() const noexcept;
        TStringBuf DarkMagenta() const noexcept;
        TStringBuf DarkCyan() const noexcept;
        TStringBuf DarkWhite() const noexcept;

        /// Compatibility; prefer using methods without `Color` suffix in their names.
        /// Note: these behave differently from their un-suffixed counterparts.
        /// While functions declared above will reset colors completely, these will only reset foreground color and
        /// style, without changing the background color and underline/italic settings. Also, names of these functions
        /// don't conform with standard, e.g. `YellowColor` actually emits the `lite yellow` escape code.
        TStringBuf OldColor() const noexcept;
        TStringBuf BoldColor() const noexcept;
        TStringBuf BlackColor() const noexcept;
        TStringBuf BlueColor() const noexcept;
        TStringBuf GreenColor() const noexcept;
        TStringBuf CyanColor() const noexcept;
        TStringBuf RedColor() const noexcept;
        TStringBuf PurpleColor() const noexcept;
        TStringBuf BrownColor() const noexcept;
        TStringBuf LightGrayColor() const noexcept;
        TStringBuf DarkGrayColor() const noexcept;
        TStringBuf LightBlueColor() const noexcept;
        TStringBuf LightGreenColor() const noexcept;
        TStringBuf LightCyanColor() const noexcept;
        TStringBuf LightRedColor() const noexcept;
        TStringBuf LightPurpleColor() const noexcept;
        TStringBuf YellowColor() const noexcept;
        TStringBuf WhiteColor() const noexcept;

        inline bool IsTTY() const noexcept {
            return IsTTY_;
        }

        inline void SetIsTTY(bool value) noexcept {
            IsTTY_ = value;
        }

        inline void Enable() noexcept {
            SetIsTTY(true);
        }

        inline void Disable() noexcept {
            SetIsTTY(false);
        }

    private:
        bool IsTTY_;
    };

    /// Singletone `TColors` instances for stderr/stdout.
    TColors& StdErr();
    TColors& StdOut();

    /// Choose `TColors` depending on output stream. If passed stream is stderr/stdout, return a corresponding
    /// singletone. Otherwise, return a disabled singletone (which you can, but should *not* enable).
    TColors& AutoColors(IOutputStream& os);

    /// Calculate total length of all ANSI escape codes in the text.
    size_t TotalAnsiEscapeCodeLen(TStringBuf text);
}

TStringBuf ToStringBuf(NColorizer::EAnsiCode x);
TString ToString(NColorizer::EAnsiCode x);
