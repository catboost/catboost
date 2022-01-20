#pragma once

#include "node.h"

#include <library/cpp/yson/consumer.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

class TNodeVisitor
{
public:
    TNodeVisitor(NYson::IYsonConsumer* consumer, bool sortMapKeys = false);

    void Visit(const TNode& node);
    void VisitMap(const TNode::TMapType& nodeMap);
    void VisitList(const TNode::TListType& nodeMap);

private:
    NYson::IYsonConsumer* Consumer_;
    bool SortMapKeys_;

private:
    void VisitAny(const TNode& node);

    void VisitString(const TNode& node);
    void VisitInt64(const TNode& node);
    void VisitUint64(const TNode& node);
    void VisitDouble(const TNode& node);
    void VisitBool(const TNode& node);
    void VisitEntity();
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
