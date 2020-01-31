#pragma once

#include "decomposition_table.h"

#include <util/charset/unidata.h>
#include <util/charset/wide.h>
#include <util/generic/hash.h>
#include <util/generic/vector.h>
#include <util/generic/algorithm.h>
#include <util/generic/singleton.h>
#include <util/generic/noncopyable.h>
#include <utility>

namespace NUnicode {
    enum ENormalization {
        NFD,
        NFC,
        NFKD,
        NFKC,
    };

    // Грубо говоря:
    // NFD расскладывает "ё" на "е + диакритику"
    // NFC сначала всё раскладывает, потом всё что может - складывает
    // NFKD делает то же, что и NFD. Кроме того, например, римскую IV (\x2163)
    //     превращает в латинские I и V
    // NFKC - NFKD + композиция (римская четвёрка из I и V, естественно, не образуется)

    // Формальная спецификация: http://www.unicode.org/reports/tr15/

    namespace NPrivate {
        inline const wchar32* Decomposition(const TDecompositionTable& table, wchar32 ch) {
            return table.Get(ch, static_cast<const wchar32*>(nullptr));
        }

        class TDecompositor {
        private:
            const TDecompositionTable& Table;

        public:
            inline TDecompositor(const TDecompositionTable& table)
                : Table(table)
            {
            }

            inline const wchar32* Decomposition(wchar32 ch) const {
                return NPrivate::Decomposition(Table, ch);
            }
        };

        template <bool IsCompat>
        struct TStandartDecompositor: public TDecompositor {
            TStandartDecompositor()
                : TDecompositor(NPrivate::DecompositionTable<IsCompat>())
            {
            }
        };

        template <ENormalization N>
        struct TShift;

        template <>
        struct TShift<NFD> {
            static const WC_TYPE Value = NFD_QC;
        };
        template <>
        struct TShift<NFC> {
            static const WC_TYPE Value = NFC_QC;
        };
        template <>
        struct TShift<NFKD> {
            static const WC_TYPE Value = NFKD_QC;
        };
        template <>
        struct TShift<NFKC> {
            static const WC_TYPE Value = NFKC_QC;
        };

        template <ENormalization N>
        inline bool Normalized(wchar32 ch) {
            return CharInfo(ch) & NPrivate::TShift<N>::Value;
        }

        class TComposition {
        private:
            struct TRawData {
                wchar32 Lead;
                wchar32 Tail;
                wchar32 Comp;
            };

            static const TRawData RawData[];
            static const size_t RawDataSize;

            class TKey: public std::pair<wchar32, wchar32> {
            public:
                inline TKey(wchar32 a, wchar32 b)
                    : std::pair<wchar32, wchar32>(a, b)
                {
                }

                inline size_t Hash() const {
                    return CombineHashes(first, second);
                }
            };

            template <class T>
            struct THash {
                inline size_t operator()(const T& t) const {
                    return t.Hash();
                }
            };

            typedef THashMap<TKey, wchar32, THash<TKey>> TData;
            TData Data;

        public:
            TComposition();

            inline wchar32 Composite(wchar32 lead, wchar32 tail) const {
                TData::const_iterator i = Data.find(TKey(lead, tail));
                if (i == Data.end())
                    return 0;

                return i->second;
            }
        };

        typedef std::pair<wchar32, TCombining> TSymbol;
        typedef TVector<TSymbol> TBuffer;

        template <bool doCompose>
        class TCompositor;

        template <>
        class TCompositor<false> {
        public:
            inline void DoComposition(TBuffer& buffer) {
                Y_UNUSED(buffer);
            }
        };

        template <>
        class TCompositor<true> {
        private:
            static const wchar32 NonComposite = 0;
            const TComposition* Composition;

        public:
            inline TCompositor()
                : Composition(Singleton<TComposition>())
            {
            }

            inline void DoComposition(TBuffer& buffer) {
                if (buffer.size() < 2)
                    return;

                const TSymbol& leadSymbol = buffer[0];
                if (leadSymbol.second != 0)
                    return;

                wchar32 lead = leadSymbol.first;
                bool oneMoreTurnPlease = false;
                do {
                    oneMoreTurnPlease = false;
                    TCombining lastCombining = 0;
                    for (TBuffer::iterator i = buffer.begin() + 1, mi = buffer.end(); i != mi; ++i) {
                        TCombining currentCombining = i->second;
                        if (!(currentCombining != lastCombining && currentCombining != 0 || lastCombining == 0 && currentCombining == 0))
                            continue;

                        lastCombining = currentCombining;
                        wchar32 comb = Composition->Composite(lead, i->first);
                        if (comb == NonComposite)
                            continue;

                        lead = comb;
                        buffer.erase(i);
                        oneMoreTurnPlease = true;
                        break;
                    }
                } while (oneMoreTurnPlease);

                Y_ASSERT(DecompositionCombining(lead) == 0);
                buffer[0] = TSymbol(lead, 0);
            }
        };

        template <ENormalization N, typename TCharType>
        inline bool Normalized(const TCharType* begin, const TCharType* end) {
            TCombining lastCanonicalClass = 0;
            for (const TCharType* i = begin; i != end;) {
                wchar32 ch = ReadSymbolAndAdvance(i, end);

                TCombining canonicalClass = DecompositionCombining(ch);
                if (lastCanonicalClass > canonicalClass && canonicalClass != 0)
                    return false;

                if (!Normalized<N>(ch))
                    return false;

                lastCanonicalClass = canonicalClass;
            }
            return true;
        }
    }

    template <bool compat>
    inline const wchar32* Decomposition(wchar32 ch) {
        return NPrivate::Decomposition(NPrivate::DecompositionTable<compat>(), ch);
    }

    template <ENormalization N, class TDecompositor = NPrivate::TDecompositor>
    class TNormalizer : NNonCopyable::TNonCopyable {
    private:
        static const ENormalization Norm = N;
        static const bool IsCompat = Norm == NFKD || Norm == NFKC;
        static const bool RequireComposition = Norm == NFC || Norm == NFKC;

        typedef NPrivate::TSymbol TSymbol;
        typedef NPrivate::TBuffer TBuffer;

        TBuffer Buffer;

        NPrivate::TCompositor<RequireComposition> Compositor;
        const TDecompositor& Decompositor;

    private:
        static inline bool Compare(const TSymbol& a, const TSymbol& b) {
            return a.second < b.second;
        }

        struct TComparer {
            inline bool operator()(const TSymbol& a, const TSymbol& b) {
                return Compare(a, b);
            }
        };

        template <class T>
        static inline void Write(const TBuffer::const_iterator& begin, const TBuffer::const_iterator& end, T& out) {
            for (TBuffer::const_iterator i = begin; i != end; ++i) {
                WriteSymbol(i->first, out);
            }
        }

        static inline void Write(const TBuffer::const_iterator& begin, const TBuffer::const_iterator& end, TUtf32String& out) {  // because WriteSymbol from util/charset/wide.h works wrong in this case
            for (TBuffer::const_iterator i = begin; i != end; ++i) {
                out += i->first;
            }
        }

        inline void SortBuffer() {
            if (Buffer.size() < 2)
                return;

            StableSort(Buffer.begin(), Buffer.end(), TComparer());
        }

        template <class T>
        inline void AddCharNoDecomposition(wchar32 c, T& out) {
            TCombining cc = DecompositionCombining(c);
            if (cc == 0) {
                SortBuffer();
                Buffer.push_back(TBuffer::value_type(c, cc));

                Compositor.DoComposition(Buffer);

                if (Buffer.size() > 1) {
                    Write(Buffer.begin(), Buffer.end() - 1, out);
                    Buffer.erase(Buffer.begin(), Buffer.end() - 1); // TODO I don't like this
                }
            } else {
                Buffer.push_back(TBuffer::value_type(c, cc));
            }
        }

        template <class T>
        inline void AddChar(wchar32 c, T& out) {
            const wchar32* decompBegin = Decompositor.Decomposition(c);
            if (decompBegin) {
                while (*decompBegin) {
                    Y_ASSERT(Decompositor.Decomposition(*decompBegin) == nullptr);
                    AddCharNoDecomposition(*(decompBegin++), out);
                }
                return;
            } else {
                AddCharNoDecomposition(c, out);
            }
        }

        template <class T, typename TCharType>
        inline void DoNormalize(const TCharType* begin, const TCharType* end, T& out) {
            Buffer.clear();

            for (const TCharType* i = begin; i != end;) {
                AddChar(ReadSymbolAndAdvance(i, end), out);
            }

            SortBuffer();
            Compositor.DoComposition(Buffer);
            Write(Buffer.begin(), Buffer.end(), out);
        }

    public:
        TNormalizer()
            : Decompositor(*Singleton<NPrivate::TStandartDecompositor<IsCompat>>())
        {
        }

        TNormalizer(const TDecompositor& decompositor)
            : Decompositor(decompositor)
        {
        }

        template <class T, typename TCharType>
        inline void Normalize(const TCharType* begin, const TCharType* end, T& out) {
            if (NPrivate::Normalized<Norm>(begin, end)) {
                for (const TCharType* i = begin; i != end; ++i) {
                    WriteSymbol(*i, out);
                }
            } else {
                DoNormalize(begin, end, out);
            }
        }

        template <typename TCharType>
        inline void Normalize(const TCharType* begin, const TCharType* end, TUtf32String& out) {
            if (NPrivate::Normalized<Norm>(begin, end)) {
                for (const TCharType* i = begin; i != end;) {
                    out += ReadSymbolAndAdvance(i, end);
                }
            } else {
                DoNormalize(begin, end, out);
            }
        }

        template <class T, typename TCharType>
        inline void Normalize(const TCharType* begin, size_t len, T& out) {
            return Normalize(begin, begin + len, out);
        }

        template <typename TCharType>
        inline TBasicString<TCharType> Normalize(const TBasicString<TCharType>& src) {
            if (NPrivate::Normalized<Norm>(src.begin(), src.end())) {
                // nothing to normalize
                return src;
            } else {
                TBasicString<TCharType> res;
                res.reserve(src.length());
                DoNormalize(src.begin(), src.end(), res);
                return res;
            }
        }
    };
}

//! decompose utf16 or utf32 string to any container supporting push_back or to T*
template <NUnicode::ENormalization Norm, class T, typename TCharType>
inline void Normalize(const TCharType* begin, size_t len, T& out) {
    ::NUnicode::TNormalizer<Norm> dec;
    dec.Normalize(begin, len, out);
}

template <NUnicode::ENormalization N, typename TCharType>
inline TBasicString<TCharType> Normalize(const TCharType* str, size_t len) {
    TBasicString<TCharType> res;
    res.reserve(len);

    Normalize<N>(str, len, res);

    return res;
}

template <NUnicode::ENormalization N, typename TCharType>
inline TBasicString<TCharType> Normalize(const TBasicString<TCharType>& str) {
    ::NUnicode::TNormalizer<N> dec;
    return dec.Normalize(str);
}

template <NUnicode::ENormalization N, typename TCharType>
inline TBasicString<TCharType> Normalize(const TBasicStringBuf<TCharType> str) {
    return Normalize<N>(str.data(), str.size());
}
