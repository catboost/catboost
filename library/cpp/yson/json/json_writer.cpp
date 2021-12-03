#include "json_writer.h"

#include <library/cpp/json/json_writer.h>

namespace NYT {
    ////////////////////////////////////////////////////////////////////////////////

    static bool IsSpecialJsonKey(const TStringBuf& key) {
        return key.size() > 0 && key[0] == '$';
    }

    ////////////////////////////////////////////////////////////////////////////////

    TJsonWriter::TJsonWriter(
        IOutputStream* output,
        ::NYson::EYsonType type,
        EJsonFormat format,
        EJsonAttributesMode attributesMode,
        ESerializedBoolFormat booleanFormat)
        : TJsonWriter(
            output,
            NJson::TJsonWriterConfig{}.SetFormatOutput(format == JF_PRETTY),
            type,
            attributesMode,
            booleanFormat
        )
    {}

    TJsonWriter::TJsonWriter(
        IOutputStream* output,
        NJson::TJsonWriterConfig config,
        ::NYson::EYsonType type,
        EJsonAttributesMode attributesMode,
        ESerializedBoolFormat booleanFormat)
        : Output(output)
        , Type(type)
        , AttributesMode(attributesMode)
        , BooleanFormat(booleanFormat)
        , Depth(0)
    {
        if (Type == ::NYson::EYsonType::MapFragment) {
            ythrow ::NYson::TYsonException() << ("Map fragments are not supported by Json");
        }

        UnderlyingJsonWriter.Reset(new NJson::TJsonWriter(
            output,
            config));
        JsonWriter = UnderlyingJsonWriter.Get();
        HasAttributes = false;
        InAttributesBalance = 0;
    }

    void TJsonWriter::EnterNode() {
        if (AttributesMode == JAM_NEVER) {
            HasAttributes = false;
        } else if (AttributesMode == JAM_ON_DEMAND) {
            // Do nothing
        } else if (AttributesMode == JAM_ALWAYS) {
            if (!HasAttributes) {
                JsonWriter->OpenMap();
                JsonWriter->Write("$attributes");
                JsonWriter->OpenMap();
                JsonWriter->CloseMap();
            }
            HasAttributes = true;
        }
        HasUnfoldedStructureStack.push_back(HasAttributes);

        if (HasAttributes) {
            JsonWriter->Write("$value");
            HasAttributes = false;
        }

        Depth += 1;
    }

    void TJsonWriter::LeaveNode() {
        Y_ASSERT(!HasUnfoldedStructureStack.empty());
        if (HasUnfoldedStructureStack.back()) {
            // Close map of the {$attributes, $value}
            JsonWriter->CloseMap();
        }
        HasUnfoldedStructureStack.pop_back();

        Depth -= 1;

        if (Depth == 0 && Type == ::NYson::EYsonType::ListFragment && InAttributesBalance == 0) {
            JsonWriter->Flush();
            Output->Write("\n");
        }
    }

    bool TJsonWriter::IsWriteAllowed() {
        if (AttributesMode == JAM_NEVER) {
            return InAttributesBalance == 0;
        }
        return true;
    }

    void TJsonWriter::OnStringScalar(TStringBuf value) {
        if (IsWriteAllowed()) {
            EnterNode();
            WriteStringScalar(value);
            LeaveNode();
        }
    }

    void TJsonWriter::OnInt64Scalar(i64 value) {
        if (IsWriteAllowed()) {
            EnterNode();
            JsonWriter->Write(value);
            LeaveNode();
        }
    }

    void TJsonWriter::OnUint64Scalar(ui64 value) {
        if (IsWriteAllowed()) {
            EnterNode();
            JsonWriter->Write(value);
            LeaveNode();
        }
    }

    void TJsonWriter::OnDoubleScalar(double value) {
        if (IsWriteAllowed()) {
            EnterNode();
            JsonWriter->Write(value);
            LeaveNode();
        }
    }

    void TJsonWriter::OnBooleanScalar(bool value) {
        if (IsWriteAllowed()) {
            if (BooleanFormat == SBF_STRING) {
                OnStringScalar(value ? "true" : "false");
            } else {
                EnterNode();
                JsonWriter->Write(value);
                LeaveNode();
            }
        }
    }

    void TJsonWriter::OnEntity() {
        if (IsWriteAllowed()) {
            EnterNode();
            JsonWriter->WriteNull();
            LeaveNode();
        }
    }

    void TJsonWriter::OnBeginList() {
        if (IsWriteAllowed()) {
            EnterNode();
            JsonWriter->OpenArray();
        }
    }

    void TJsonWriter::OnListItem() {
    }

    void TJsonWriter::OnEndList() {
        if (IsWriteAllowed()) {
            JsonWriter->CloseArray();
            LeaveNode();
        }
    }

    void TJsonWriter::OnBeginMap() {
        if (IsWriteAllowed()) {
            EnterNode();
            JsonWriter->OpenMap();
        }
    }

    void TJsonWriter::OnKeyedItem(TStringBuf name) {
        if (IsWriteAllowed()) {
            if (IsSpecialJsonKey(name)) {
                WriteStringScalar(TString("$") + name);
            } else {
                WriteStringScalar(name);
            }
        }
    }

    void TJsonWriter::OnEndMap() {
        if (IsWriteAllowed()) {
            JsonWriter->CloseMap();
            LeaveNode();
        }
    }

    void TJsonWriter::OnBeginAttributes() {
        InAttributesBalance += 1;
        if (AttributesMode != JAM_NEVER) {
            JsonWriter->OpenMap();
            JsonWriter->Write("$attributes");
            JsonWriter->OpenMap();
        }
    }

    void TJsonWriter::OnEndAttributes() {
        InAttributesBalance -= 1;
        if (AttributesMode != JAM_NEVER) {
            HasAttributes = true;
            JsonWriter->CloseMap();
        }
    }

    void TJsonWriter::WriteStringScalar(const TStringBuf& value) {
        JsonWriter->Write(value);
    }

    void TJsonWriter::Flush() {
        JsonWriter->Flush();
    }

    ////////////////////////////////////////////////////////////////////////////////

}
