#pragma once

#include "util.h"

namespace NTextProcessing::NDictionary {
    constexpr int INVALID_POSITION = -2;
    constexpr int FIRST_POSITION = -1;

    using TTokenId = ui32;
    using TPair = std::pair<TTokenId, TTokenId>;

    struct TPairStat {
        ui64 Count = 0;
        TPair Pair = {0, 0};
        TVector<std::pair<int, int>> Positions;

        bool operator<(const TPairStat& other) const {
            return std::tie(
                Count,
                Min(other.Pair.first, other.Pair.second),
                Max(other.Pair.first, other.Pair.second),
                other.Pair.first
            ) < std::tie(
                other.Count,
                Min(Pair.first, Pair.second),
                Max(Pair.first, Pair.second),
                Pair.first
            );
        }
    };

    template <class TokenIdType>
    class TEraseList {
    private:
        struct TElement {
            TokenIdType Token;
            int Left;
            int Right;
        };

        TVector<TElement> TokenIds;

    public:
        TEraseList() = default;

        TEraseList(const TVector<TokenIdType>& tokens) {
            TokenIds.reserve(tokens.size());
            for (size_t i = 0; i < tokens.size(); i++) {
                TokenIds.emplace_back(TElement{tokens[i], (int)i - 1, (int)i + 1});
            }
        }

        bool Empty() const {
            return TokenIds.empty();
        }

        int Size() const {
            return TokenIds.size();
        }

        void PushToken(TokenIdType token) {
            TokenIds.emplace_back(TElement{token, (int)TokenIds.size() - 1, (int)TokenIds.size() + 1});
        }

        void UpdateToken(int position, TokenIdType token) {
            TokenIds[position].Token = token;
        }

        bool IsValidElement(int position) const {
            return TokenIds[position].Left != INVALID_POSITION;
        }

        bool IsFirstElement(int position) const {
            return TokenIds[position].Left == FIRST_POSITION;
        }

        bool IsLastElement(int position) const {
            return TokenIds[position].Right == (int)TokenIds.size();
        }

        int GetPrevPosition(int position) const {
            return TokenIds[position].Left;
        }

        int GetNextPosition(int position) const {
            return TokenIds[position].Right;
        }

        std::pair<TokenIdType, TokenIdType> GetPair(int position) const {
            Y_ASSERT(!IsLastElement(position));
            return {TokenIds[position].Token, TokenIds[TokenIds[position].Right].Token};
        }

        void Erase(int position) {
            int prevPosition = TokenIds[position].Left;
            int nextPosition = TokenIds[position].Right;
            if (prevPosition >= 0) {
                TokenIds[prevPosition].Right = nextPosition;
            }
            if (nextPosition < (int)TokenIds.size()) {
                TokenIds[nextPosition].Left = prevPosition;
            }
            TokenIds[position].Left = INVALID_POSITION;
        }

        TVector<TokenIdType> GetValidElements() const {
            TVector<TokenIdType> result;
            for (size_t i = 0; i < TokenIds.size(); ++i) {
                if (IsValidElement(i)) {
                    result.push_back(TokenIds[i].Token);
                }
            }
            return result;
        }
    };
}
