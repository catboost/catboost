#pragma once

#include <library/cpp/dbg_output/dump.h>

#include <util/generic/array_ref.h>
#include <util/generic/xrange.h>
#include <util/stream/output.h>


namespace NCB {

    namespace NPrivate {

        template <class T>
        struct TDbgDumpWithIndices {
            inline TDbgDumpWithIndices(TConstArrayRef<T> v, bool onSeparateLines)
                : V(v)
                , OnSeparateLines(onSeparateLines)
            {}

            inline void DumpTo(IOutputStream* out) const {
                (*out) << '[';
                if (V.size() > 0) {
                    if (OnSeparateLines) {
                        (*out) << Endl;
                    }
                    for (auto i : xrange<size_t>(V.size())) {
                        (*out) << (OnSeparateLines ? "\t" : (i == 0 ? "" : ", ")) << i << ':'
                            << ::DbgDump(V[i]);
                        if (OnSeparateLines) {
                            (*out) << Endl;
                        }
                    }
                }
                (*out) << ']';
                if (OnSeparateLines) {
                    (*out) << Endl;
                }
            }

        private:
            TConstArrayRef<T> V;
            bool OnSeparateLines;
        };

        template <class T>
        static inline IOutputStream& operator<<(IOutputStream& out, const TDbgDumpWithIndices<T>& d) {
            d.DumpTo(&out);
            return out;
        }

    }

    template <class T>
    static inline NPrivate::TDbgDumpWithIndices<T> DbgDumpWithIndices(
        TConstArrayRef<T> v,
        bool onSeparateLines = false
    ) {
        return NPrivate::TDbgDumpWithIndices<T>(v, onSeparateLines);
    }

    template <class T>
    static inline NPrivate::TDbgDumpWithIndices<T> DbgDumpWithIndices(
        const TVector<T>& v,
        bool onSeparateLines = false
    ) {
        return NPrivate::TDbgDumpWithIndices<T>(v, onSeparateLines);
    }
}
