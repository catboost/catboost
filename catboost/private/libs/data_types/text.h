#pragma once

#include <util/generic/algorithm.h>
#include <util/generic/map.h>
#include <util/generic/vector.h>
#include <util/generic/ylimits.h>
#include <util/stream/output.h>
#include <util/system/yassert.h>
#include <util/ysaveload.h>


namespace NCB {

    struct TTokenId {
        ui32 Id;
        static constexpr ui32 ILLEGAL_TOKEN_ID = Max<ui32>();

        TTokenId()
            : Id(ILLEGAL_TOKEN_ID) {}

        TTokenId(ui32 id)
            : Id(id) {}

        operator ui32() const {
            return Id;
        }

        bool operator==(ui32 rhs) const {
            return Id == rhs;
        }

        bool operator!=(ui32 rhs) const {
            return !(rhs == *this);
        }

        bool operator==(const TTokenId& rhs) const {
            return Id == rhs.Id;
        }

        bool operator!=(const TTokenId& rhs) const {
            return !(rhs == *this);
        }

        bool operator<(const TTokenId& rhs) const {
            return Id < rhs.Id;
        }

        bool operator>(const TTokenId& rhs) const {
            return rhs < *this;
        }

        bool operator<=(const TTokenId& rhs) const {
            return !(rhs < *this);
        }

        bool operator>=(const TTokenId& rhs) const {
            return !(*this < rhs);
        }

        Y_SAVELOAD_DEFINE(Id);
    };

    class TText {
    private:
        class TTokenToCountPair {
        public:
            using TValueType = ui32;

            TTokenToCountPair(TTokenId tokenId, TValueType count)
                : TokenId(tokenId)
                , Counter(count)
            {}

            explicit TTokenToCountPair(std::pair<TTokenId, TValueType> pair)
                : TokenId(pair.first)
                , Counter(pair.second)
            {}

            TTokenId Token() const {
                return TokenId;
            }

            TValueType Count() const {
                return Counter;
            }

            void IncreaseCount() {
                Counter++;
            }

            bool operator==(const TTokenToCountPair& rhs) const {
                return std::tie(TokenId, Counter) == std::tie(rhs.TokenId, rhs.Counter);
            }
            bool operator!=(const TTokenToCountPair& rhs) const {
                return !(rhs == *this);
            }

        private:
            TTokenId TokenId;
            TValueType Counter;
        };

        using TTokenToCountStorage = TVector<TTokenToCountPair>;

        class TTokenToCountIterator {
        public:
            explicit TTokenToCountIterator(const TText& text, ui32 position = 0)
                : OwnerText(text)
                , CurrentIndex(position)
            {}

            const TTokenToCountPair* operator->() const {
                Y_ASSERT(CurrentIndex < OwnerText.TokenToCount.size());
                return &OwnerText.TokenToCount[CurrentIndex];
            }
            const TTokenToCountPair& operator*() const {
                Y_ASSERT(CurrentIndex < OwnerText.TokenToCount.size());
                return OwnerText.TokenToCount[CurrentIndex];
            }

            TTokenToCountIterator& operator++() {
                CurrentIndex++;
                return *this;
            }
            const TTokenToCountIterator operator++(int) {
                TTokenToCountIterator tmp(*this);
                operator++();
                return tmp;
            }
            bool operator==(const TTokenToCountIterator& rhs) const {
                return (OwnerText == rhs.OwnerText) && (CurrentIndex == rhs.CurrentIndex);
            }
            bool operator!=(const TTokenToCountIterator& rhs) const {
                return !(rhs == *this);
            }

        private:
            const TText& OwnerText;
            ui32 CurrentIndex;

            friend class TText;
        };

    public:
        using TCountType = TTokenToCountPair::TValueType;

        TText() : TokenToCount() {}

        TText(const TMap<TTokenId, TTokenToCountPair::TValueType>& tokenToCount) {
            for (const auto& [token, count]: tokenToCount) {
                TokenToCount.push_back(TTokenToCountPair{token, count});
            }
        }

        TText(TVector<ui32>&& tokenIds) {
            Sort(tokenIds);
            for (const auto& tokenId : tokenIds) {
                if (TokenToCount.empty() || TokenToCount.back().Token() != tokenId) {
                    TokenToCount.push_back(TTokenToCountPair{TTokenId(tokenId), 1});
                } else {
                    TokenToCount.back().IncreaseCount();
                }
            }
        }

        TTokenToCountIterator begin() const {
            return TTokenToCountIterator(*this);
        }

        TTokenToCountIterator end() const {
            return TTokenToCountIterator(*this, TokenToCount.size());
        }

        bool operator==(const TText& rhs) const {
            return TokenToCount == rhs.TokenToCount;
        }

        bool operator!=(const TText& rhs) const {
            return !(*this == rhs);
        }

        void Clear() {
            TokenToCount.clear();
        }

        TTokenToCountIterator Find(TTokenId tokenId) const {
            // TODO(d-kruchinin): binary search
            for (auto iterator = begin(); iterator != end(); ++iterator) {
                if (iterator->Token() == tokenId) {
                    return iterator;
                }
            }
            return end();
        }

    private:
        TTokenToCountStorage TokenToCount;
    };
}

template <>
inline void Out<NCB::TText>(IOutputStream& stream, const NCB::TText& text) {
    for (const auto& tokenToCount : text) {
        stream << "TokenId=" << static_cast<ui32>(tokenToCount.Token())
            << ", Count=" << tokenToCount.Count() << Endl;
    }
}
