#pragma once

#include <cstddef>

namespace NCompactTrie {
    class ILeafSkipper;

    enum TDirection {
        D_LEFT,
        D_FINAL,
        D_NEXT,
        D_RIGHT,
        D_MAX
    };

    inline TDirection& operator++(TDirection& direction) {
        direction = static_cast<TDirection>(direction + 1);
        return direction;
    }

    inline TDirection& operator--(TDirection& direction) {
        direction = static_cast<TDirection>(direction - 1);
        return direction;
    }

    class TNode {
    public:
        TNode();
        // Processes epsilon links and sets ForwardOffset to correct value. Assumes an epsilon link doesn't point to an epsilon link.
        TNode(const char* data, size_t offset, const ILeafSkipper& skipFunction);

        size_t GetOffset() const {
            return Offset;
        }

        size_t GetLeafOffset() const {
            return Offsets[D_FINAL];
        }
        size_t GetLeafLength() const {
            return LeafLength;
        }
        size_t GetCoreLength() const {
            return CoreLength;
        }

        size_t GetOffsetByDirection(TDirection direction) const {
            return Offsets[direction];
        }

        size_t GetForwardOffset() const {
            return Offsets[D_NEXT];
        }
        size_t GetLeftOffset() const {
            return Offsets[D_LEFT];
        }
        size_t GetRightOffset() const {
            return Offsets[D_RIGHT];
        }
        char GetLabel() const {
            return Label;
        }

        bool IsFinal() const {
            return GetLeafOffset() != 0;
        }

        bool HasEpsilonLinkForward() const {
            return GetForwardOffset() > Offset + CoreLength;
        }

    private:
        size_t Offsets[D_MAX];
        size_t Offset;
        size_t LeafLength;
        size_t CoreLength;

        char Label;
    };

}
