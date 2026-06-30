#pragma once

#include <library/cpp/yson/consumer.h>

#include <library/cpp/json/json_reader.h>

#include <util/generic/stack.h>

namespace NYT {
    class TYson2JsonCallbacksAdapter
       : public NJson::TJsonCallbacks {
    public:
        class TState {
        private:
            // Stores current context stack
            // If true - we are in a list
            // If false - we are in a map
            TStack<bool> ContextStack;

            friend class TYson2JsonCallbacksAdapter;
        };

    public:
        TYson2JsonCallbacksAdapter(
            ::NYson::TYsonConsumerBase* impl,
            bool throwException = false,
            ui64 maxDepth = std::numeric_limits<ui64>::max());

        bool OnNull() override;
        bool OnBoolean(bool val) override;
        bool OnInteger(long long val) override;
        bool OnUInteger(unsigned long long val) override;
        bool OnString(const TStringBuf& val) override;
        bool OnDouble(double val) override;
        bool OnOpenArray() override;
        bool OnCloseArray() override;
        bool OnOpenMap() override;
        bool OnCloseMap() override;
        bool OnMapKey(const TStringBuf& val) override;

        TState State() const {
            return State_;
        }

        void Reset(const TState& state) {
            State_ = state;
        }

    private:
        void WrapIfListItem();

    private:
        ::NYson::TYsonConsumerBase* Impl_;
        TState State_;
        ui64 MaxDepth_;
    };
}
