#include "node.h"
#include "leaf_skipper.h"
#include "comptrie_impl.h"

#include <util/system/yassert.h>
#include <util/generic/yexception.h>

namespace NCompactTrie {
    TNode::TNode()
        : Offset(0)
        , LeafLength(0)
        , CoreLength(0)
        , Label(0)
    {
        for (auto& offset : Offsets) {
            offset = 0;
        }
    }

    // We believe that epsilon links are found only on the forward-position and that afer jumping an epsilon link you come to an ordinary node.

    TNode::TNode(const char* data, size_t offset, const ILeafSkipper& skipFunction)
        : Offset(offset)
        , LeafLength(0)
        , CoreLength(0)
        , Label(0)
    {
        for (auto& anOffset : Offsets) {
            anOffset = 0;
        }
        if (!data)
            return; // empty constructor

        const char* datapos = data + offset;
        char flags = *(datapos++);
        Y_ASSERT(!IsEpsilonLink(flags));
        Label = *(datapos++);

        size_t leftsize = LeftOffsetLen(flags);
        size_t& leftOffset = Offsets[D_LEFT];
        leftOffset = UnpackOffset(datapos, leftsize);
        if (leftOffset) {
            leftOffset += Offset;
        }
        datapos += leftsize;

        size_t rightsize = RightOffsetLen(flags);
        size_t& rightOffset = Offsets[D_RIGHT];
        rightOffset = UnpackOffset(datapos, rightsize);
        if (rightOffset) {
            rightOffset += Offset;
        }
        datapos += rightsize;

        if (flags & MT_FINAL) {
            Offsets[D_FINAL] = datapos - data;
            LeafLength = skipFunction.SkipLeaf(datapos);
        }

        CoreLength = 2 + leftsize + rightsize + LeafLength;
        if (flags & MT_NEXT) {
            size_t& forwardOffset = Offsets[D_NEXT];
            forwardOffset = Offset + CoreLength;
            // There might be an epsilon link at the forward position.
            // If so, set the ForwardOffset to the value that points to the link's end.
            const char* forwardPos = data + forwardOffset;
            const char forwardFlags = *forwardPos;
            if (IsEpsilonLink(forwardFlags)) {
                // Jump through the epsilon link.
                size_t epsilonOffset = UnpackOffset(forwardPos + 1, forwardFlags & MT_SIZEMASK);
                if (!epsilonOffset) {
                    ythrow yexception() << "Corrupted epsilon link";
                }
                forwardOffset += epsilonOffset;
            }
        }
    }

}
