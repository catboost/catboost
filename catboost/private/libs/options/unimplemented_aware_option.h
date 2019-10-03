#pragma once

#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/option.h>

#include <util/system/types.h>

namespace NCatboostOptions {
    template <ETaskType Type>
    struct TTaskFlag;

#define DECLARE_TASK_FLAG(TaskFlag, i)        \
    template <>                               \
    struct TTaskFlag<TaskFlag> {              \
        static const ui64 Flag = 1ULL << (i); \
    };

    DECLARE_TASK_FLAG(ETaskType::GPU, 0)
    DECLARE_TASK_FLAG(ETaskType::CPU, 1)
//    For more flags:
//    DECLARE_TASK_FLAG(ETaskType::MasterSlave, 2)
//    DECLARE_TASK_FLAG(ETaskType::..., 3)
#undef DECLARE_TASK_FLAG

    template <ETaskType... Types>
    struct TSupportedTasks;

    template <>
    struct TSupportedTasks<> {
        static constexpr ui64 Flags() {
            return 0;
        }
    };

    template <ETaskType Type,
              ETaskType... RestTypes>
    struct TSupportedTasks<Type, RestTypes...> {
        static constexpr ui64 Flags() {
            return TTaskFlag<Type>::Flag | TSupportedTasks<RestTypes...>::Flags();
        }

        template <ETaskType Task>
        static constexpr bool IsSupported() {
            return static_cast<bool>(TTaskFlag<Task>::Flag & Flags());
        }

        static bool IsSupported(ETaskType taskType) {
            switch (taskType) {
                case ETaskType::GPU: {
                    return IsSupported<ETaskType::GPU>();
                }
                case ETaskType::CPU: {
                    return IsSupported<ETaskType::CPU>();
                }
                default: {
                    ythrow TCatBoostException() << "Unknown task type " << taskType;
                }
            }
        }
    };

    template <class TValue,
              class TSupportedTasks>
    class TUnimplementedAwareOption: public TOption<TValue> {
    public:
        TUnimplementedAwareOption(const TString& key,
                                  const TValue& defaultValue,
                                  ETaskType taskType,
                                  ELoadUnimplementedPolicy policy = ELoadUnimplementedPolicy::ExceptionOnChange)
            : TOption<TValue>(key, defaultValue)
            , TaskType(taskType)
            , LoadUnimplementedPolicy(policy)
        {
        }

        template <ETaskType Task>
        static constexpr bool IsUnimplemented() {
            return !TSupportedTasks::template IsSupported<Task>();
        }

        bool IsUnimplementedForCurrentTask() const {
            return !TSupportedTasks::IsSupported(TaskType);
        }

        void SetTaskType(ETaskType taskType) {
            TaskType = taskType;
        }

        const TValue& Get() const override {
            //if all options were templates, we could use static_assert and compile-time check
            bool isUnimplemented = IsUnimplementedForCurrentTask();
            CB_ENSURE(!isUnimplemented, "Option " << TOption<TValue>::GetName() << " is unimplemented for task " << TaskType);
            return TOption<TValue>::Get();
        }

        const TValue& GetUnchecked() const {
            return TOption<TValue>::Get();
        }

        TValue& GetUnchecked() {
            return TOption<TValue>::Get();
        }

        TValue& Get() override {
            //if all options were templates, we could use static_assert and compile-time check
            bool isUnimplemented = IsUnimplementedForCurrentTask();
            Y_ASSERT(!isUnimplemented);
            CB_ENSURE(!isUnimplemented, "Option " << TOption<TValue>::GetName() << " is unimplemented for task " << TaskType);
            return TOption<TValue>::Get();
        }

        ETaskType GetCurrentTaskType() const {
            return TaskType;
        }

        ELoadUnimplementedPolicy GetLoadUnimplementedPolicy() const {
            return LoadUnimplementedPolicy;
        }

        void ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy policy) {
            LoadUnimplementedPolicy = policy;
        }

        inline TUnimplementedAwareOption& operator=(const TValue& value) {
            TOption<TValue>::Set(value);
            return *this;
        }

        bool operator==(const TUnimplementedAwareOption& rhs) const {
            return TOption<TValue>::operator==(static_cast<const TOption<TValue>&>(rhs));
        }

        bool operator!=(const TUnimplementedAwareOption& rhs) const {
            return !(rhs == *this);
        }

        template <typename TComparableType>
        bool operator==(const TComparableType& otherValue) const {
            return TOption<TValue>::operator==(otherValue);
        }

        template <typename TComparableType>
        bool operator!=(const TComparableType& otherValue) const {
            return TOption<TValue>::operator!=(otherValue);
        }

    private:
        ETaskType TaskType;
        ELoadUnimplementedPolicy LoadUnimplementedPolicy;
    };

    template <class TValue>
    using TGpuOnlyOption = TUnimplementedAwareOption<TValue, TSupportedTasks<ETaskType::GPU>>;

    template <class TValue>
    using TCpuOnlyOption = TUnimplementedAwareOption<TValue, TSupportedTasks<ETaskType::CPU>>;
}
