#include <library/cpp/testing/unittest/registar.h>
#include <catboost/cuda/cuda_lib/serialization/task_factory.h>
#include <catboost/cuda/cuda_lib/tasks_impl/kernel_task.h>

using namespace NCudaLib;

struct TSampleKernel: public NKernelHost::TStatelessKernel {
    TVector<ui32> Data1;
    ui32 Data2 = 0;

    void Run(const TCudaStream& stream) {
        Y_UNUSED(stream);
    }

    Y_SAVELOAD_DEFINE(Data1, Data2);
};

template <class T>
struct TSampleKernel2: public NKernelHost::TStatelessKernel {
    int Data1 = 0;
    T Data2 = 100500;

    void Run(const TCudaStream& stream) {
        Y_UNUSED(stream);
    }

    Y_SAVELOAD_DEFINE(Data1, Data2);
};

REGISTER_KERNEL(100500, TSampleKernel);
REGISTER_KERNEL_TEMPLATE(10050042, TSampleKernel2, float);
REGISTER_KERNEL_TEMPLATE(10050043, TSampleKernel2, int);

Y_UNIT_TEST_SUITE(TSerializationTest) {
    template <class T>
    void TestTemplateKernel(T val2) {
        using TKernel = TSampleKernel2<T>;
        TKernel kernel;
        kernel.Data1 = 11;
        kernel.Data2 = val2;

        using TCmd = TGpuKernelTask<TKernel>;
        TCmd cmd(TKernel(kernel), 1);

        auto serialized = TTaskSerializer::Serialize(cmd);
        auto deserialized = TTaskSerializer::LoadCommand(serialized);
        auto deserializedPtr = dynamic_cast<TCmd*>(deserialized.Get());

        UNIT_ASSERT_UNEQUAL(deserializedPtr, nullptr);
        UNIT_ASSERT_EQUAL(deserializedPtr->GetStreamId(), 1u);
        UNIT_ASSERT_EQUAL(deserializedPtr->GetCommandType(), EComandType::StreamKernel);
        UNIT_ASSERT_EQUAL(deserializedPtr->GetKernel().Data1, 11);
        UNIT_ASSERT_EQUAL(deserializedPtr->GetKernel().Data2, val2);
    }

    Y_UNIT_TEST(TestKernelSerialization) {
        {
            TSampleKernel kernel;
            kernel.Data1.push_back(0u);
            kernel.Data1.push_back(10u);
            kernel.Data2 = 10;

            using TCmd = TGpuKernelTask<TSampleKernel>;
            TCmd cmd(TSampleKernel(kernel), 10);
            auto serialized = TTaskSerializer::Serialize(cmd);
            auto deserialized = TTaskSerializer::LoadCommand(serialized);
            auto deserializedPtr = dynamic_cast<TCmd*>(deserialized.Get());

            UNIT_ASSERT_UNEQUAL(deserializedPtr, nullptr);
            UNIT_ASSERT_EQUAL(deserializedPtr->GetStreamId(), 10u);
            UNIT_ASSERT_EQUAL(deserializedPtr->GetCommandType(), EComandType::StreamKernel);
            UNIT_ASSERT_EQUAL(deserializedPtr->GetKernel().Data1, kernel.Data1);
            UNIT_ASSERT_EQUAL(deserializedPtr->GetKernel().Data2, kernel.Data2);
        }

        TestTemplateKernel<float>(10.5);
        TestTemplateKernel<int>(-100);
    }
}
