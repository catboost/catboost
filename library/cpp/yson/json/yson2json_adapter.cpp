#include "yson2json_adapter.h"

namespace NYT {
    TYson2JsonCallbacksAdapter::TYson2JsonCallbacksAdapter(
        ::NYson::TYsonConsumerBase* impl,
        bool throwException,
        ui64 maxDepth)
        : NJson::TJsonCallbacks(throwException)
        , Impl_(impl)
        , MaxDepth_(maxDepth)
    {
    }

    bool TYson2JsonCallbacksAdapter::OnNull() {
        WrapIfListItem();
        Impl_->OnEntity();
        return true;
    }

    bool TYson2JsonCallbacksAdapter::OnBoolean(bool val) {
        WrapIfListItem();
        Impl_->OnBooleanScalar(val);
        return true;
    }

    bool TYson2JsonCallbacksAdapter::OnInteger(long long val) {
        WrapIfListItem();
        Impl_->OnInt64Scalar(val);
        return true;
    }

    bool TYson2JsonCallbacksAdapter::OnUInteger(unsigned long long val) {
        WrapIfListItem();
        Impl_->OnUint64Scalar(val);
        return true;
    }

    bool TYson2JsonCallbacksAdapter::OnString(const TStringBuf& val) {
        WrapIfListItem();
        Impl_->OnStringScalar(val);
        return true;
    }

    bool TYson2JsonCallbacksAdapter::OnDouble(double val) {
        WrapIfListItem();
        Impl_->OnDoubleScalar(val);
        return true;
    }

    bool TYson2JsonCallbacksAdapter::OnOpenArray() {
        WrapIfListItem();
        State_.ContextStack.push(true);
        if (State_.ContextStack.size() > MaxDepth_) {
            return false;
        }
        Impl_->OnBeginList();
        return true;
    }

    bool TYson2JsonCallbacksAdapter::OnCloseArray() {
        State_.ContextStack.pop();
        Impl_->OnEndList();
        return true;
    }

    bool TYson2JsonCallbacksAdapter::OnOpenMap() {
        WrapIfListItem();
        State_.ContextStack.push(false);
        if (State_.ContextStack.size() > MaxDepth_) {
            return false;
        }
        Impl_->OnBeginMap();
        return true;
    }

    bool TYson2JsonCallbacksAdapter::OnCloseMap() {
        State_.ContextStack.pop();
        Impl_->OnEndMap();
        return true;
    }

    bool TYson2JsonCallbacksAdapter::OnMapKey(const TStringBuf& val) {
        Impl_->OnKeyedItem(val);
        return true;
    }

    void TYson2JsonCallbacksAdapter::WrapIfListItem() {
        if (!State_.ContextStack.empty() && State_.ContextStack.top()) {
            Impl_->OnListItem();
        }
    }
}
