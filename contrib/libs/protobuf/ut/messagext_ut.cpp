#include <library/cpp/unittest/registar.h>

#include <contrib/libs/protobuf/message.h>
#include <contrib/libs/protobuf/messagext.h>
#include <contrib/libs/protobuf/descriptor.h>
#include <contrib/libs/protobuf/descriptor_database.h>
#include <contrib/libs/protobuf/compiler/importer.h>
#include <contrib/libs/protobuf/text_format.h>

#include <util/generic/vector.h>
#include <util/generic/string.h>
#include <util/stream/input.h>
#include <util/stream/output.h>
#include <util/stream/str.h>
#include <util/stream/file.h>
#include <util/system/tempfile.h>
#include <util/random/random.h>
#include <util/folder/dirut.h>

#include <contrib/libs/protobuf/ut/test.pb.h>
#include <contrib/libs/protobuf/ut/ns.pb.h>

using namespace google::protobuf::io;

Y_UNIT_TEST_SUITE(TProtobufTest) {
    static void GenerateStrings(int n, TVector<TString>& strings) {
        for (int i = 0; i < n; ++i) {
            unsigned int lengthMax = 2 << (RandomNumber<unsigned int>() % 15);
            unsigned int length = RandomNumber<unsigned int>() % lengthMax;
            strings.emplace_back(length, 'a');
        }
    }

    Y_UNIT_TEST(TestOutput) {
        TTestMessage test;
        test.Setff("42");
        TStringStream out;
        out << test;
        UNIT_ASSERT_STRINGS_EQUAL(out.Str(), "{ ff: \"42\" }");
    }

    Y_UNIT_TEST(TestUtf8Output) {
        TTestMessage test;
        test.Setff("Яндекс");
        TStringStream out;
        out << test.ShortUtf8DebugString();
        UNIT_ASSERT_STRINGS_EQUAL(out.Str(), "ff: \"Яндекс\"");
    }

    Y_UNIT_TEST(TestNamespacedOutput) {
        NMyPackage::TEmpty empty;
        TStringStream out;
        out << empty;
        UNIT_ASSERT_STRINGS_EQUAL(out.Str(), "{  }");
    }

    Y_UNIT_TEST(TestReadWriteSeq) {
        TVector<TString> strings;
        GenerateStrings(1000, strings);

        TString s;
        TStringOutput o(s);
        TCopyingOutputStreamAdaptor oo(&o);

        for (TVector<TString>::const_iterator it = strings.begin(); it != strings.end(); ++it) {
            TTestMessage m;
            m.Setff(*it);
            UNIT_ASSERT(SerializeToZeroCopyStreamSeq(&m, &oo));
        }

        oo.Flush();

        //Cerr << "size of s is " << s.length() << Endl;

        TStringInput i(s);
        TCopyingInputStreamAdaptor ii(&i);

        for (TVector<TString>::const_iterator it = strings.begin(); it != strings.end(); ++it) {
            TTestMessage m;
            UNIT_ASSERT(ParseFromZeroCopyStreamSeq(&m, &ii));
            UNIT_ASSERT_VALUES_EQUAL(*it, m.Getff());
        }

        TTestMessage n;
        UNIT_ASSERT(!ParseFromZeroCopyStreamSeq(&n, &ii));
    }

    Y_UNIT_TEST(TestAsStreamSeq) {
        TTestMessage m;
        m.Setff("REVIEW: NOW");

        TString oldSchool;
        {
            // For an unknown reason (in presence of ::Save),
            // a popular oldschool method - with a tendency to forget flushing the adaptor
            TStringOutput o(oldSchool);
            TCopyingOutputStreamAdaptor oo(&o);
            UNIT_ASSERT(SerializeToZeroCopyStreamSeq(&m, &oo));
        }
        TString newWave;
        TStringOutput newWaveOutput(newWave);
        // As simple as that
        newWaveOutput << m.AsStreamSeq();
        UNIT_ASSERT_STRINGS_EQUAL(oldSchool, newWave);
    };

    Y_UNIT_TEST(TestSaveLoad) {
        TVector<TString> strings;
        GenerateStrings(1000, strings);

        TStringStream s;

        for (TVector<TString>::const_iterator it = strings.begin(); it != strings.end(); ++it) {
            TTestMessage m;
            m.Setff(*it);
            UNIT_ASSERT_NO_EXCEPTION(::Save(&s, m));
        }

        for (TVector<TString>::const_iterator it = strings.begin(); it != strings.end(); ++it) {
            TTestMessage m;
            //UNIT_ASSERT_NO_EXCEPTION(::Load(&s, m));
            ::Load(&s, m);
            UNIT_ASSERT_VALUES_EQUAL(*it, m.Getff());
        }

        TTestMessage n;
        UNIT_ASSERT_EXCEPTION(::Load(&s, n), yexception);
    }

    class TFakeError {};
    class TFakeOutput: public IOutputStream {
    private:
        void DoWrite(const void* , size_t ) {
            throw TFakeError();
        }
    };

    class TFakeInput: public IInputStream {
    private:
        size_t DoRead(void* , size_t) {
            throw TFakeError();
        }
    };

    Y_UNIT_TEST(TestErrorOnWrite1) {
        TStringStream bigString;
        for (size_t i = 0; i < 100000; ++i) {
            bigString << "test";
        }

        TTestMessage testMessage;
        testMessage.Setff(bigString.Str());

        try {
            TFakeOutput stream;
            UNIT_ASSERT(!testMessage.SerializeToStream(&stream));
        } catch (...) {
            UNIT_ASSERT(false);
        }
    }

    Y_UNIT_TEST(TestErrorOnWrite2) {
        TTestMessage testMessage;
        testMessage.Setff("test");
        try {
            TFakeOutput stream;
            UNIT_ASSERT(!testMessage.SerializeToStream(&stream));
        } catch (...) {
            UNIT_ASSERT(false);
        }
    }

    Y_UNIT_TEST(TestErrorOnRead1) {
        google::protobuf::LogHandler* old = google::protobuf::SetLogHandler(NULL);
        TTestMessage testMessage;
        try {
            TFakeInput stream;
            UNIT_ASSERT(!testMessage.ParseFromStream(&stream));
        } catch (...) {
            UNIT_ASSERT(false);
        }
        google::protobuf::SetLogHandler(old);
    }
    Y_UNIT_TEST(TestErrorOnRead2) {
        TTestMessage2 testMessage;
        try {
            TFakeInput stream;
            UNIT_ASSERT(!testMessage.ParseFromStream(&stream));
        } catch (...) {
            UNIT_ASSERT(false);
        }
    }

    // google/protobuf/google/protobuf/descriptor.proto + proto read from disk
    class TBuilder {
    public:
        TBuilder()
            : SourceTree()
            , DiskFiles(&SourceTree)
            , Builtins()
            , Merged(&Builtins, &DiskFiles)
            , Pool(&Merged, DiskFiles.GetValidationErrorCollector())
        {
            SourceTree.MapPath("", NFs::CurrentWorkingDirectory());

            const TString name = "google/protobuf/descriptor.proto";
            const NProtoBuf::FileDescriptor* file = NProtoBuf::DescriptorPool::generated_pool()->FindFileByName(name);
            if (!file)
                ythrow yexception() << "Cannot find " << name << " in generated_pool";

            NProtoBuf::FileDescriptorProto fileProto;
            file->CopyTo(&fileProto);
            Builtins.Add(fileProto);
        }

        const NProtoBuf::FileDescriptor* BuildFileDescriptor(const TString& name) const {
            return Pool.FindFileByName(name);
        }

    private:
        NProtoBuf::compiler::DiskSourceTree SourceTree;
        NProtoBuf::compiler::SourceTreeDescriptorDatabase DiskFiles;
        NProtoBuf::SimpleDescriptorDatabase Builtins;
        NProtoBuf::protobuf::MergedDescriptorDatabase Merged;

        NProtoBuf::DescriptorPool Pool;
    };

    static inline TString SerializeFileDescriptor(const NProtoBuf::FileDescriptor* file) {
        NProtoBuf::FileDescriptorProto proto;
        file->CopyTo(&proto);
        return proto.SerializeAsString();
    }

    Y_UNIT_TEST(TestCustomOptions) {
        const TString protobuf =
            "import \"google/protobuf/descriptor.proto\";\n"
            "option optimize_for = CODE_SIZE;\n"
            "message FooOptions { optional int32 opt1 = 1; optional string opt2 = 2; }\n"
            "extend google.protobuf.FileOptions { optional string my_file_option = 50000; }\n"
            "extend google.protobuf.FileOptions { optional string my_file_option2 = 50001; }\n"
            "extend google.protobuf.MessageOptions { optional int32 my_message_option = 50002; }\n"
            "extend google.protobuf.FieldOptions { optional FooOptions my_field_option = 50003; }\n"
            "option (my_file_option) = \"123\";\n"
            "option (my_file_option2) = \"456\";\n"
            "message TestMessage {\n"
            "    option (my_message_option) = 42;\n"
            "    required string text = 1 [default = \"123\"];\n"
            "    repeated int32 samples = 2 [packed=true];\n"
            "    optional int32 old_field = 3 [deprecated=true];\n"
            "    optional int32 xxx = 4 [(my_field_option) = { opt1: 123 opt2:\"baz\"}];\n"
            "}";

        const TString fname = "test_options.proto";
        const TTempFile tempFile("./" + fname);
        {
            TOFStream out(tempFile.Name());
            out << protobuf << Endl;
        }

        TString binaryDescr;
        {
            TBuilder builder;
            const NProtoBuf::FileDescriptor* fileDescr = builder.BuildFileDescriptor(fname);
            UNIT_ASSERT(fileDescr != NULL);
            binaryDescr = SerializeFileDescriptor(fileDescr);

            TOFStream out(tempFile.Name());
            out << fileDescr->DebugString() << Endl;
        }
        {
            TBuilder builder;
            const NProtoBuf::FileDescriptor* fileDescr = builder.BuildFileDescriptor(fname);
            UNIT_ASSERT(fileDescr != NULL);
            UNIT_ASSERT(SerializeFileDescriptor(fileDescr) == binaryDescr);
        }
    }

    Y_UNIT_TEST(TestJSON) {
        TJSONTest message;
        message.AddA(1);
        message.AddA(2);
        message.AddB()->Setff("test");
        message.SetTheC("yandex\n");

        static const char* EXPECTED = "{\"A\":[1,2],\"B\":[{\"ff\":\"test\"}],\"TheC\":\"yandex\\n\"}";

        TStringStream out;
        out << message.AsJSON();

        UNIT_ASSERT_STRINGS_EQUAL(out.Str(), EXPECTED);

        TStringStream out2;
        out2 << ((google::protobuf::Message&)message).AsJSON();
        UNIT_ASSERT_STRINGS_EQUAL(out2.Str(), EXPECTED);
    }

    Y_UNIT_TEST(TestAsBinary) {
        TJSONTest proto;
        proto.SetTheC("c");

        TString canonic;
        proto.SerializeToString(&canonic);

        TStringStream stream;
        proto.SerializeToStream(&stream);

        TStringStream simple;
        simple << proto.AsBinary();

        UNIT_ASSERT_STRINGS_EQUAL(canonic, simple.Str());
        UNIT_ASSERT_STRINGS_EQUAL(stream.Str(), simple.Str());
    }

    Y_UNIT_TEST(TestParseTextFormat) {
        class TNullErrorCollector : public google::protobuf::io::ErrorCollector {
            void AddError(int , int , const google::protobuf::string& ) override {}
        } nullErrorCollector;

        TString text = "ff: \"value\"; unknown: \"unkvalue\"";
        {
            TString text= "unknown: \"value\"";
            TTestMessage msg;
            google::protobuf::TextFormat::Parser parser;
            parser.RecordErrorsTo(&nullErrorCollector);
            parser.AllowUnknownField(false);
            UNIT_ASSERT_EQUAL(parser.ParseFromString(text, &msg), false);
        }
        {
            TTestMessage msg;
            google::protobuf::TextFormat::Parser parser;
            parser.RecordErrorsTo(&nullErrorCollector);
            parser.AllowUnknownField(true);
            UNIT_ASSERT_EQUAL(parser.ParseFromString(text, &msg), true);
            UNIT_ASSERT_EQUAL(msg.Getff(), "value");
        }
        {
            TTestMessage msg;
            google::protobuf::TextFormat::Parser parser;
            parser.RecordErrorsTo(&nullErrorCollector);
            parser.AllowUnknownField(true);

            const TString unknownIntFieldText = 
                "ff: \"value_1\"\n"
                "unknownIntField: 12345\n"
                "unknownIntFieldComma: 12345,\n"
                "unknownIntFieldSemicolon: 12345,\n";
            UNIT_ASSERT_EQUAL(parser.ParseFromString(unknownIntFieldText, &msg), true);
            UNIT_ASSERT_EQUAL(msg.Getff(), "value_1");

            const TString unknownStringFieldText = 
                "unknownStringField: \"unknown_value\"\n"
                "ff: \"value_2\"\n"
                "unknownStringFieldComma: \"unknown_value\",\n"
                "unknownStringFieldSemicolon: \"unknown_value\";\n";
            UNIT_ASSERT_EQUAL(parser.ParseFromString(unknownStringFieldText, &msg), true);
            UNIT_ASSERT_EQUAL(msg.Getff(), "value_2");

            const TString unknownMinusInfFieldText = 
                "unknownInfField: -inf\n"
                "unknownInfFieldComma: -inf,\n"
                "ff: \"value_3\"\n"
                "unknownInfFieldSemicolon: -inf;\n";
            UNIT_ASSERT_EQUAL(parser.ParseFromString(unknownMinusInfFieldText, &msg), true);
            UNIT_ASSERT_EQUAL(msg.Getff(), "value_3");

            const TString unknownTypeIdentifierField = 
                "unknownTypeIdentifierField: TYPE_IDENTIFIER\n"
                "unknownTypeIdentifierFieldComma: TYPE_IDENTIFIER,\n"
                "unknownTypeIdentifierFieldSemicolon: TYPE_IDENTIFIER;\n"
                "ff: \"value_4\"\n";

            UNIT_ASSERT_EQUAL(parser.ParseFromString(unknownTypeIdentifierField, &msg), true);
            UNIT_ASSERT_EQUAL(msg.Getff(), "value_4");

            const TString unknownMessage = 
                "UnknownMessageSemicolon {\n"
                "   UnknownFieldSemicolon: \"unknown_value\";\n" 
                "};\n"
                "\n"
                "ff: \"value_5\"\n";

            UNIT_ASSERT_EQUAL(parser.ParseFromString(unknownMessage, &msg), true);
            UNIT_ASSERT_EQUAL(msg.Getff(), "value_5");
        }
    }
}
