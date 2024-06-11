#ifndef FORMAT_ANALYSER_INL_H_
#error "Direct inclusion of this file is not allowed, include format_analyser.h"
// For the sake of sane code completion.
#include "format_analyser.h"
#endif

#include "format_arg.h"

namespace NYT::NDetail {

////////////////////////////////////////////////////////////////////////////////

consteval bool Contains(std::string_view sv, char symbol)
{
    return sv.find(symbol) != std::string_view::npos;
}

template <class... TArgs>
consteval void TFormatAnalyser::ValidateFormat(std::string_view format)
{
    std::array<std::string_view, sizeof...(TArgs)> markers = {};
    std::array<TSpecifiers, sizeof...(TArgs)> specifiers{GetSpecifiers<TArgs>()...};

    int markerCount = 0;
    int currentMarkerStart = -1;

    for (int index = 0; index < std::ssize(format); ++index) {
        auto symbol = format[index];

        // Parse verbatim text.
        if (currentMarkerStart == -1) {
            if (symbol == IntroductorySymbol) {
                // Marker maybe begins.
                currentMarkerStart = index;
            }
            continue;
        }

        // NB: We check for %% first since
        // in order to verify if symbol is a specifier
        // we need markerCount to be within range of our
        // specifier array.
        if (symbol == IntroductorySymbol) {
            if (currentMarkerStart + 1 != index) {
                // '%a% detected'
                CrashCompilerWrongTermination("You may not terminate flag sequence other than %% with \'%\' symbol");
                return;
            }
            // '%%' detected --- skip
            currentMarkerStart = -1;
            continue;
        }

        // We are inside of marker.
        if (markerCount == std::ssize(markers)) {
            // To many markers
            CrashCompilerNotEnoughArguments("Number of arguments supplied to format is smaller than the number of flag sequences");
            return;
        }

        if (Contains(specifiers[markerCount].Conversion, symbol)) {
            // Marker has finished.

            markers[markerCount]
                = std::string_view(format.begin() + currentMarkerStart, index - currentMarkerStart + 1);
            currentMarkerStart = -1;
            ++markerCount;

            continue;
        }

        if (!Contains(specifiers[markerCount].Flags, symbol)) {
            CrashCompilerWrongFlagSpecifier("Symbol is not a valid flag specifier; See FlagSpecifiers");
        }
    }

    if (currentMarkerStart != -1) {
        // Runaway marker.
        CrashCompilerMissingTermination("Unterminated flag sequence detected; Use \'%%\' to type plain %");
        return;
    }

    if (markerCount < std::ssize(markers)) {
        // Missing markers.
        CrashCompilerTooManyArguments("Number of arguments supplied to format is greater than the number of flag sequences");
        return;
    }

    // TODO(arkady-e1ppa): Consider per-type verification
    // of markers.
}

template <class TArg>
consteval auto TFormatAnalyser::GetSpecifiers()
{
    if constexpr (!CFormattable<TArg>) {
        CrashCompilerNotFormattable<TArg>("Your specialization of TFormatArg is broken");
    }

    return TSpecifiers{
        .Conversion = std::string_view{
            std::data(TFormatArg<TArg>::ConversionSpecifiers),
            std::size(TFormatArg<TArg>::ConversionSpecifiers)},
        .Flags = std::string_view{
            std::data(TFormatArg<TArg>::FlagSpecifiers),
            std::size(TFormatArg<TArg>::FlagSpecifiers)},
    };
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NDetail
