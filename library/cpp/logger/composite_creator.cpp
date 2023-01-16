#include "composite_creator.h"
#include "composite.h"
#include "uninitialized_creator.h"

THolder<TLogBackend> TCompositeBackendCreator::DoCreateLogBackend() const {
    auto res = MakeHolder<TCompositeLogBackend>();
    for (const auto& child : Children) {
        res->AddLogBackend(child->CreateLogBackend());
    }
    return std::move(res);
}


TCompositeBackendCreator::TCompositeBackendCreator()
    : TLogBackendCreatorBase("composite")
{}

bool TCompositeBackendCreator::Init(const IInitContext& ctx) {
    for (const auto& child : ctx.GetChildren("SubLogger")) {
        Children.emplace_back(MakeHolder<TLogBackendCreatorUninitialized>());
        if (!Children.back()->Init(*child)) {
            return false;
        }
    }
    return true;
}

ILogBackendCreator::TFactory::TRegistrator<TCompositeBackendCreator> TCompositeBackendCreator::Registrar("composite");

void TCompositeBackendCreator::DoToJson(NJson::TJsonValue& value) const {
    for (const auto& child: Children) {
        child->ToJson(value["SubLogger"].AppendValue(NJson::JSON_MAP));
    }
}
