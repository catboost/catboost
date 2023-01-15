#pragma once

#include "node.h"

#include <library/cpp/json/json_reader.h>

#include <library/cpp/yson/consumer.h>

#include <util/generic/stack.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

class TNodeBuilder
    : public TYsonConsumerBase
{
public:
    TNodeBuilder(TNode* node);

    void OnStringScalar(const TStringBuf&) override;
    void OnInt64Scalar(i64) override;
    void OnUint64Scalar(ui64) override;
    void OnDoubleScalar(double) override;
    void OnBooleanScalar(bool) override;
    void OnEntity() override;
    void OnBeginList() override;
    void OnListItem() override;
    void OnEndList() override;
    void OnBeginMap() override;
    void OnKeyedItem(const TStringBuf&) override;
    void OnEndMap() override;
    void OnBeginAttributes() override;
    void OnEndAttributes() override;
    void OnNode(TNode node);

private:
    TStack<TNode*> Stack_;

private:
    inline void AddNode(TNode node, bool pop);
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
