#pragma once

#include "engine.h"
#include <library/cpp/colorizer/output.h>

#ifndef DBG_OUTPUT_DEFAULT_COLOR_SCHEME
#define DBG_OUTPUT_DEFAULT_COLOR_SCHEME NDbgDump::NColorScheme::TPlain
#endif

#define DBG_OUTPUT_COLOR_HANDLER(NAME) \
    template <class S>                 \
    inline void NAME(S& stream)

namespace NDbgDump {
    namespace NColorScheme {
        /// Start by copying this one if you want to define a custom color scheme.
        struct TPlain {
            // Foreground color modifiers
            DBG_OUTPUT_COLOR_HANDLER(Markup) {
                Y_UNUSED(stream);
            }
            DBG_OUTPUT_COLOR_HANDLER(String) {
                Y_UNUSED(stream);
            }
            DBG_OUTPUT_COLOR_HANDLER(Literal) {
                Y_UNUSED(stream);
            }
            DBG_OUTPUT_COLOR_HANDLER(ResetType) {
                Y_UNUSED(stream);
            }

            // Background color modifiers
            DBG_OUTPUT_COLOR_HANDLER(Key) {
                Y_UNUSED(stream);
            }
            DBG_OUTPUT_COLOR_HANDLER(Value) {
                Y_UNUSED(stream);
            }
            DBG_OUTPUT_COLOR_HANDLER(ResetRole) {
                Y_UNUSED(stream);
            }
        };

        /// Use this one if you want colors but are lazy enough to define a custom color scheme.
        /// Be careful enough to use DumpRaw for avoiding an endless recursion.
        /// Enforce controls whether colors should be applied even if stdout is not a TTY.
        template <bool Enforce = false>
        class TEyebleed {
        public:
            TEyebleed() {
                if (Enforce) {
                    Colors.Enable();
                }
            }

            // Foreground color modifiers
            DBG_OUTPUT_COLOR_HANDLER(Markup) {
                stream << DumpRaw(Colors.LightGreenColor());
            }
            DBG_OUTPUT_COLOR_HANDLER(String) {
                stream << DumpRaw(Colors.YellowColor());
            }
            DBG_OUTPUT_COLOR_HANDLER(Literal) {
                stream << DumpRaw(Colors.LightRedColor());
            }
            DBG_OUTPUT_COLOR_HANDLER(ResetType) {
                stream << DumpRaw(Colors.OldColor());
            }

            // Background color modifiers
            // TODO: support backgrounds in library/cpp/colorizer
            DBG_OUTPUT_COLOR_HANDLER(Key) {
                if (Depth++ == 0 && Colors.IsTTY()) {
                    stream << DumpRaw(TStringBuf("\033[42m"));
                }
            }
            DBG_OUTPUT_COLOR_HANDLER(Value) {
                if (Depth++ == 0 && Colors.IsTTY()) {
                    stream << DumpRaw(TStringBuf("\033[44m"));
                }
            }
            DBG_OUTPUT_COLOR_HANDLER(ResetRole) {
                if (--Depth == 0 && Colors.IsTTY()) {
                    stream << DumpRaw(TStringBuf("\033[49m"));
                }
            }

        private:
            NColorizer::TColors Colors;
            size_t Depth = 0;
        };
    }
}

namespace NPrivate {
    template <typename CS>
    struct TColorSchemeContainer {
        CS ColorScheme;
    };
}
