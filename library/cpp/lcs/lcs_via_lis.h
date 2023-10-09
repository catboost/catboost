#pragma once

#include <library/cpp/containers/paged_vector/paged_vector.h>

#include <util/generic/ptr.h>
#include <util/generic/hash.h>
#include <util/generic/vector.h>
#include <util/generic/algorithm.h>
#include <util/memory/pool.h>

namespace NLCS {
    template <typename TVal>
    struct TLCSCtx {
        typedef TVector<ui32> TSubsequence;
        typedef THashMap<TVal, TSubsequence, THash<TVal>, TEqualTo<TVal>, ::TPoolAllocator> TEncounterIndex;
        typedef TVector<std::pair<ui32, ui32>> TLastIndex;
        typedef NPagedVector::TPagedVector<TSubsequence, 4096> TCover;

        TMemoryPool Pool;
        THolder<TEncounterIndex> Encounters;
        TLastIndex LastIndex;
        TCover Cover;

        TSubsequence ResultBuffer;

        TLCSCtx()
            : Pool(16 * 1024 - 64, TMemoryPool::TExpGrow::Instance())
        {
            Reset();
        }

        void Reset() {
            Encounters.Reset(nullptr);
            Pool.Clear();
            Encounters.Reset(new TEncounterIndex(&Pool));
            LastIndex.clear();
            Cover.clear();
            ResultBuffer.clear();
        }
    };

    namespace NPrivate {
        template <typename TIt, typename TVl>
        struct TSequence {
            typedef TIt TIter;
            typedef TVl TVal;

            const TIter Begin;
            const TIter End;
            const size_t Size;

            TSequence(TIter beg, TIter end)
                : Begin(beg)
                , End(end)
                , Size(end - beg)
            {
            }
        };

        template <typename TVal, typename TSequence, typename TResult>
        size_t MakeLCS(TSequence s1, TSequence s2, TResult* res = nullptr, TLCSCtx<TVal>* ctx = nullptr) {
            typedef TLCSCtx<TVal> TCtx;

            THolder<TCtx> ctxhld;

            if (!ctx) {
                ctxhld.Reset(new TCtx());
                ctx = ctxhld.Get();
            } else {
                ctx->Reset();
            }

            size_t maxsize = Max(s1.Size, s2.Size);
            auto& index = *(ctx->Encounters);

            for (auto it = s1.Begin; it != s1.End; ++it) {
                index[*it];
            }

            for (auto it = s2.Begin; it != s2.End; ++it) {
                auto hit = index.find(*it);

                if (hit != index.end()) {
                    hit->second.push_back(it - s2.Begin);
                }
            }

            if (!res) {
                auto& lastindex = ctx->ResultBuffer;
                lastindex.reserve(maxsize);

                for (auto it1 = s1.Begin; it1 != s1.End; ++it1) {
                    const auto& sub2 = index[*it1];

                    for (auto it2 = sub2.rbegin(); it2 != sub2.rend(); ++it2) {
                        ui32 x = *it2;

                        auto lit = LowerBound(lastindex.begin(), lastindex.end(), x);

                        if (lit == lastindex.end()) {
                            lastindex.push_back(x);
                        } else {
                            *lit = x;
                        }
                    }
                }

                return lastindex.size();
            } else {
                auto& lastindex = ctx->LastIndex;
                auto& cover = ctx->Cover;

                lastindex.reserve(maxsize);

                for (auto it1 = s1.Begin; it1 != s1.End; ++it1) {
                    const auto& sub2 = index[*it1];

                    for (auto it2 = sub2.rbegin(); it2 != sub2.rend(); ++it2) {
                        ui32 x = *it2;

                        auto lit = LowerBound(lastindex.begin(), lastindex.end(), std::make_pair(x, (ui32)0u));

                        if (lit == lastindex.end()) {
                            lastindex.push_back(std::make_pair(x, cover.size()));
                            cover.emplace_back();
                            cover.back().push_back(x);
                        } else {
                            *lit = std::make_pair(x, lit->second);
                            cover[lit->second].push_back(x);
                        }
                    }
                }

                if (cover.empty()) {
                    return 0;
                }

                std::reverse(cover.begin(), cover.end());

                auto& resbuf = ctx->ResultBuffer;

                resbuf.push_back(cover.front().front());

                for (auto it = cover.begin() + 1; it != cover.end(); ++it) {
                    auto pit = UpperBound(it->begin(), it->end(), resbuf.back(), std::greater<ui32>());

                    Y_ABORT_UNLESS(pit != it->end(), " ");

                    resbuf.push_back(*pit);
                }

                std::reverse(resbuf.begin(), resbuf.end());

                for (auto it = resbuf.begin(); it != resbuf.end(); ++it) {
                    res->push_back(*(s2.Begin + *it));
                }

                return lastindex.size();
            }
        }
    }

    template <typename TVal, typename TIter, typename TResult>
    size_t MakeLCS(TIter beg1, TIter end1, TIter beg2, TIter end2, TResult* res = nullptr, TLCSCtx<TVal>* ctx = nullptr) {
        typedef NPrivate::TSequence<TIter, TVal> TSeq;

        size_t sz1 = end1 - beg1;
        size_t sz2 = end2 - beg2;

        if (sz2 > sz1) {
            DoSwap(beg1, beg2);
            DoSwap(end1, end2);
            DoSwap(sz1, sz2);
        }

        return NPrivate::MakeLCS<TVal>(TSeq(beg1, end1), TSeq(beg2, end2), res, ctx);
    }

    template <typename TVal, typename TColl, typename TRes>
    size_t MakeLCS(const TColl& coll1, const TColl& coll2, TRes* res = nullptr, TLCSCtx<TVal>* ctx = nullptr) {
        return MakeLCS<TVal>(coll1.begin(), coll1.end(), coll2.begin(), coll2.end(), res, ctx);
    }

    template <typename TVal, typename TIter>
    size_t MeasureLCS(TIter beg1, TIter end1, TIter beg2, TIter end2, TLCSCtx<TVal>* ctx = nullptr) {
        return MakeLCS<TVal>(beg1, end1, beg2, end2, (TVector<TVal>*)nullptr, ctx);
    }

    template <typename TVal, typename TColl>
    size_t MeasureLCS(const TColl& coll1, const TColl& coll2, TLCSCtx<TVal>* ctx = nullptr) {
        return MeasureLCS<TVal>(coll1.begin(), coll1.end(), coll2.begin(), coll2.end(), ctx);
    }
}
