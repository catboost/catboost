#include "node_builder.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

TNodeBuilder::TNodeBuilder(TNode* node)
{
    Stack_.push(node);
}

void TNodeBuilder::OnStringScalar(TStringBuf value)
{
    AddNode(value, true);
}

void TNodeBuilder::OnInt64Scalar(i64 value)
{
    AddNode(value, true);
}

void TNodeBuilder::OnUint64Scalar(ui64 value)
{
    AddNode(value, true);
}

void TNodeBuilder::OnDoubleScalar(double value)
{
    AddNode(value, true);
}

void TNodeBuilder::OnBooleanScalar(bool value)
{
    AddNode(value, true);
}

void TNodeBuilder::OnEntity()
{
    AddNode(TNode::CreateEntity(), true);
}

void TNodeBuilder::OnBeginList()
{
    AddNode(TNode::CreateList(), false);
}

void TNodeBuilder::OnBeginList(ui64 reserveSize) {
    OnBeginList();
    Stack_.top()->AsList().reserve(reserveSize);
}

void TNodeBuilder::OnListItem()
{
    Stack_.push(&Stack_.top()->Add());
}

void TNodeBuilder::OnEndList()
{
    Stack_.pop();
}

void TNodeBuilder::OnBeginMap()
{
    AddNode(TNode::CreateMap(), false);
}

void TNodeBuilder::OnBeginMap(ui64 reserveSize) {
    OnBeginMap();
    Stack_.top()->AsMap().reserve(reserveSize);
}

void TNodeBuilder::OnKeyedItem(TStringBuf key)
{
    Stack_.push(&(*Stack_.top())[TString(key)]);
}

void TNodeBuilder::OnEndMap()
{
    Stack_.pop();
}

void TNodeBuilder::OnBeginAttributes()
{
    Stack_.push(&Stack_.top()->Attributes());
}

void TNodeBuilder::OnEndAttributes()
{
    Stack_.pop();
}

void TNodeBuilder::OnNode(TNode node)
{
    AddNode(std::move(node), true);
}

void TNodeBuilder::AddNode(TNode value, bool pop)
{
    Stack_.top()->MoveWithoutAttributes(std::move(value));
    if (pop)
        Stack_.pop();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
