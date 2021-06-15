#pragma once

#include "engine.h"
#include "dumpers.h"
#include "auto.h"
#include "colorscheme.h"

#include <util/stream/format.h>
#include <util/system/type_name.h>
#include <util/generic/hash_set.h>
#include <utility>

/*
 * Cout << DbgDump(any) << Endl;
 * Cout << DbgDumpDeep(any) << Endl;
 * Cout << DbgDump(any).SetIndent(true) << Endl;
 *
 * specialize TDumper<your type> for extending dumper
 */

namespace NPrivate {
    template <class TColorScheme>
    struct TTraitsShallow {
        struct TDump: public TDumpBase, public TColorSchemeContainer<TColorScheme> {
            template <typename... Args>
            inline TDump(Args&&... args)
                : TDumpBase(std::forward<Args>(args)...)
            {
            }

            template <class V>
            inline void Pointer(const V* v) {
                if (v) {
                    *this << DumpRaw("(") << DumpRaw(TypeName(v).data()) << DumpRaw(")") << Hex((size_t)v);
                } else {
                    *this << DumpRaw("(") << DumpRaw(TypeName<V>().data()) << DumpRaw("*)nullptr");
                }
            }
        };
    };

    template <class TColorScheme>
    struct TTraitsDeep {
        struct TDump: public TDumpBase, public TColorSchemeContainer<TColorScheme> {
            template <typename... Args>
            inline TDump(Args&&... args)
                : TDumpBase(std::forward<Args>(args)...)
            {
            }

            template <class V>
            inline void Pointer(const V* v) {
                if (v && !Visited.contains((size_t)v)) {
                    Visited.insert((size_t)v);
                    *this << DumpRaw("(") << DumpRaw(TypeName(v).data()) << DumpRaw(")") << Hex((size_t)v) << DumpRaw(" -> ") << *v;
                    Visited.erase((size_t)v);
                } else {
                    *this << DumpRaw("(") << DumpRaw(TypeName<V>().data()) << DumpRaw("*)nullptr");
                }
            }

            THashSet<size_t> Visited;
        };
    };

    template <class T, class TTraits>
    struct TDbgDump {
        inline TDbgDump(const T* t)
            : T_(t)
            , Indent(false)
        {
        }

        inline void DumpTo(IOutputStream& out) const {
            typename TTraits::TDump d(out, Indent);

            d << *T_;
        }

        inline TDbgDump& SetIndent(bool v) noexcept {
            Indent = v;

            return *this;
        }

        const T* T_;
        bool Indent;
    };

    template <class T, class TTraits>
    static inline IOutputStream& operator<<(IOutputStream& out, const TDbgDump<T, TTraits>& d) {
        d.DumpTo(out);

        return out;
    }
}

template <class T, class TColorScheme = DBG_OUTPUT_DEFAULT_COLOR_SCHEME>
static inline ::NPrivate::TDbgDump<T, ::NPrivate::TTraitsShallow<TColorScheme>> DbgDump(const T& t) {
    return {std::addressof(t)};
}

template <class T, class TColorScheme = DBG_OUTPUT_DEFAULT_COLOR_SCHEME>
static inline ::NPrivate::TDbgDump<T, ::NPrivate::TTraitsDeep<TColorScheme>> DbgDumpDeep(const T& t) {
    return {std::addressof(t)};
}
