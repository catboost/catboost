#include "proto_packer.h"

#include <library/cpp/packers/ut/test.pb.h>
#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/string.h>

using namespace NPackers;
using namespace NProtoPackerTest;

void FillRequiredFields(TTestMessage& msg) {
    msg.SetRequiredString("required_string");
    msg.SetRequiredInt32(42);
}

void FillOptionalFields(TTestMessage& msg) {
    msg.SetOptionalString("optional_string");
    msg.SetOptionalInt32(43);
}

void FillRepeatedFields(TTestMessage& msg) {
    msg.ClearRepeatedStrings();
    for (ui32 idx = 0; idx < 5; ++idx) {
        msg.AddRepeatedStrings("repeated_string" + ToString(idx));
    }
}

// do not want to use google/protobuf/util/message_differencer because of warnings
bool operator==(const TTestMessage& lhs, const TTestMessage& rhs) {
    if (lhs.GetRequiredString() != rhs.GetRequiredString() ||
        lhs.GetRequiredInt32() != rhs.GetRequiredInt32() ||
        lhs.HasOptionalString() != rhs.HasOptionalString() ||
        (lhs.HasOptionalString() && lhs.GetOptionalString() != rhs.GetOptionalString()) ||
        lhs.HasOptionalInt32() != rhs.HasOptionalInt32() ||
        (lhs.HasOptionalInt32() && lhs.GetOptionalInt32() != rhs.GetOptionalInt32()) ||
        lhs.RepeatedStringsSize() != rhs.RepeatedStringsSize())
    {
        return false;
    }
    for (ui32 idx = 0; idx < lhs.RepeatedStringsSize(); ++idx) {
        if (lhs.GetRepeatedStrings(idx) != rhs.GetRepeatedStrings(idx)) {
            return false;
        }
    }
    return true;
}

Y_UNIT_TEST_SUITE(ProtoPackerTestSuite) {
    TProtoMessagePacker<TTestMessage> Packer;
    TString Buffer;

    void DoPackUnpackTest(const TTestMessage& msg) {
        const ui32 msgByteSize = Packer.MeasureLeaf(msg);
        Buffer.resize(msgByteSize);

        Packer.PackLeaf(Buffer.begin(), msg, msgByteSize);

        TTestMessage checkMsg;
        Packer.UnpackLeaf(Buffer.begin(), checkMsg);

        UNIT_ASSERT_EQUAL(msg, checkMsg);
    }

    Y_UNIT_TEST(TestPackUnpackOnlyRequired) {
        TTestMessage msg;
        FillRequiredFields(msg);
        DoPackUnpackTest(msg);
    }

    Y_UNIT_TEST(TestPackUnpackRequiredAndOptional) {
        TTestMessage msg;
        FillRequiredFields(msg);
        FillOptionalFields(msg);
        DoPackUnpackTest(msg);
    }

    Y_UNIT_TEST(TestPackUnpackAll) {
        TTestMessage msg;
        FillRequiredFields(msg);
        FillOptionalFields(msg);
        FillRepeatedFields(msg);
        DoPackUnpackTest(msg);
    }

    Y_UNIT_TEST(TestSkipLeaf) {
        TTestMessage msgFirst;
        FillRequiredFields(msgFirst);
        TTestMessage msgSecond;
        FillRequiredFields(msgSecond);
        FillOptionalFields(msgSecond);

        const ui32 msgFirstByteSize = Packer.MeasureLeaf(msgFirst);
        const ui32 msgSecondByteSize = Packer.MeasureLeaf(msgSecond);

        Buffer.resize(msgFirstByteSize + msgSecondByteSize);
        Packer.PackLeaf(Buffer.begin(), msgFirst, msgFirstByteSize);
        Packer.PackLeaf(Buffer.begin() + msgFirstByteSize, msgSecond, msgSecondByteSize);

        TTestMessage checkMsg;
        Packer.UnpackLeaf(Buffer.begin() + Packer.SkipLeaf(Buffer.begin()), checkMsg);

        UNIT_ASSERT_EQUAL(msgSecond, checkMsg);
    }
}
