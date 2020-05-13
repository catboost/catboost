#pragma once

#include "public.h"
#include "consumer.h"

#include <library/cpp/json/json_writer.h>

#include <util/generic/vector.h>

namespace NYT {
    ////////////////////////////////////////////////////////////////////////////////

    enum EJsonFormat {
        JF_TEXT,
        JF_PRETTY
    };

    enum EJsonAttributesMode {
        JAM_NEVER,
        JAM_ON_DEMAND,
        JAM_ALWAYS
    };

    enum ESerializedBoolFormat {
        SBF_BOOLEAN,
        SBF_STRING
    };

    class TJsonWriter
       : public TYsonConsumerBase {
    public:
        TJsonWriter(
            IOutputStream* output,
            EYsonType type = YT_NODE,
            EJsonFormat format = JF_TEXT,
            EJsonAttributesMode attributesMode = JAM_ON_DEMAND,
            ESerializedBoolFormat booleanFormat = SBF_STRING);

        TJsonWriter(
            IOutputStream* output,
            NJson::TJsonWriterConfig config,
            EYsonType type = YT_NODE,
            EJsonAttributesMode attributesMode = JAM_ON_DEMAND,
            ESerializedBoolFormat booleanFormat = SBF_STRING);

        void Flush();

        void OnStringScalar(const TStringBuf& value) override;
        void OnInt64Scalar(i64 value) override;
        void OnUint64Scalar(ui64 value) override;
        void OnDoubleScalar(double value) override;
        void OnBooleanScalar(bool value) override;

        void OnEntity() override;

        void OnBeginList() override;
        void OnListItem() override;
        void OnEndList() override;

        void OnBeginMap() override;
        void OnKeyedItem(const TStringBuf& key) override;
        void OnEndMap() override;

        void OnBeginAttributes() override;
        void OnEndAttributes() override;

    private:
        THolder<NJson::TJsonWriter> UnderlyingJsonWriter;
        NJson::TJsonWriter* JsonWriter;
        IOutputStream* Output;
        EYsonType Type;
        EJsonAttributesMode AttributesMode;
        ESerializedBoolFormat BooleanFormat;

        void WriteStringScalar(const TStringBuf& value);

        void EnterNode();
        void LeaveNode();
        bool IsWriteAllowed();

        TVector<bool> HasUnfoldedStructureStack;
        int InAttributesBalance;
        bool HasAttributes;
        int Depth;
    };

    ////////////////////////////////////////////////////////////////////////////////

}
