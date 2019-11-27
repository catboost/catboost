#include "writeable_node.h"
#include "node.h"
#include "comptrie_impl.h"

namespace NCompactTrie {
    TWriteableNode::TWriteableNode()
        : LeafPos(nullptr)
        , LeafLength(0)
        , ForwardOffset(NPOS)
        , LeftOffset(NPOS)
        , RightOffset(NPOS)
        , Label(0)
    {
    }

    static size_t GetOffsetFromEnd(const TNode& node, size_t absOffset) {
        return absOffset ? absOffset - node.GetOffset() - node.GetCoreLength() : NPOS;
    }

    TWriteableNode::TWriteableNode(const TNode& node, const char* data)
        : LeafPos(node.IsFinal() ? data + node.GetLeafOffset() : nullptr)
        , LeafLength(node.GetLeafLength())
        , ForwardOffset(GetOffsetFromEnd(node, node.GetForwardOffset()))
        , LeftOffset(GetOffsetFromEnd(node, node.GetLeftOffset()))
        , RightOffset(GetOffsetFromEnd(node, node.GetRightOffset()))
        , Label(node.GetLabel())
    {
    }

    size_t TWriteableNode::Measure() const {
        size_t len = 2 + LeafLength;
        size_t fwdLen = 0;
        size_t lastLen = 0;
        size_t lastFwdLen = 0;
        // Now, increase all the offsets by the length and recalculate everything, until it converges
        do {
            lastLen = len;
            lastFwdLen = fwdLen;

            len = 2 + LeafLength;
            len += MeasureOffset(LeftOffset != NPOS ? LeftOffset + lastLen : 0);
            len += MeasureOffset(RightOffset != NPOS ? RightOffset + lastLen : 0);

            // Relative forward offset of 0 means we don't need extra length for an epsilon link.
            // But an epsilon link means we need an extra 1 for the flags and the forward offset is measured
            // from the start of the epsilon link, not from the start of our node.
            if (ForwardOffset != NPOS && ForwardOffset != 0) {
                fwdLen = MeasureOffset(ForwardOffset + lastFwdLen) + 1;
                len += fwdLen;
            }

        } while (lastLen != len || lastFwdLen != fwdLen);

        return len;
    }

    size_t TWriteableNode::Pack(char* buffer) const {
        const size_t length = Measure();

        char flags = 0;
        if (LeafPos) {
            flags |= MT_FINAL;
        }
        if (ForwardOffset != NPOS) {
            flags |= MT_NEXT;
        }

        const size_t leftOffset = LeftOffset != NPOS ? LeftOffset + length : 0;
        const size_t rightOffset = RightOffset != NPOS ? RightOffset + length : 0;
        const size_t leftOffsetSize = MeasureOffset(leftOffset);
        const size_t rightOffsetSize = MeasureOffset(rightOffset);
        flags |= (leftOffsetSize << MT_LEFTSHIFT);
        flags |= (rightOffsetSize << MT_RIGHTSHIFT);

        buffer[0] = flags;
        buffer[1] = Label;
        size_t usedLen = 2;
        usedLen += PackOffset(buffer + usedLen, leftOffset);
        usedLen += PackOffset(buffer + usedLen, rightOffset);

        if (LeafPos && LeafLength) {
            memcpy(buffer + usedLen, LeafPos, LeafLength);
            usedLen += LeafLength;
        }

        if (ForwardOffset != NPOS && ForwardOffset != 0) {
            const size_t fwdOffset = ForwardOffset + length - usedLen;
            size_t fwdOffsetSize = MeasureOffset(fwdOffset);
            buffer[usedLen++] = (char)(fwdOffsetSize & MT_SIZEMASK);
            usedLen += PackOffset(buffer + usedLen, fwdOffset);
        }
        Y_ASSERT(usedLen == length);
        return usedLen;
    }

}
