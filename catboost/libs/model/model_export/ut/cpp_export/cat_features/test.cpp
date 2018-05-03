#include <catboost/libs/model/model.h>

#include <library/unittest/registar.h>
#include <library/unittest/tests_data.h>
#include <library/unittest/env.h>
#include <library/resource/resource.h>

#include <util/generic/ymath.h>

#include <vector>

double ApplyCatboostModel(const std::vector<float>& floatFeatures, const std::vector<std::string>& catFeatures);

Y_UNIT_TEST_SUITE(CompareBinaryAndCPPModelWithCatFeatures) {
    class TCPPAndBinaryModelsComparator {
    private:
        TFullModel Calcer;

    public:
        TCPPAndBinaryModelsComparator(const void* buffer, size_t bufferLength) {
            Calcer = ReadModel(buffer, bufferLength);
        };
        bool CompareOn(const std::vector<float>& floatFeatures, const std::vector<std::string>& catFeatures) {
            TVector<TConstArrayRef<float>> floatFeaturesVec(1);
            TVector<TVector<TStringBuf>> catFeaturesVec(1, TVector<TStringBuf>(catFeatures.size()));
            floatFeaturesVec[0] = TConstArrayRef<float>(floatFeatures.data(), floatFeatures.size());
            for (size_t catFeatureIdx = 0; catFeatureIdx < catFeatures.size(); ++catFeatureIdx) {
                catFeaturesVec[0][catFeatureIdx] = catFeatures[catFeatureIdx];
            }

            double binaryModelResult = 0;
            Calcer.Calc(floatFeaturesVec, catFeaturesVec, TArrayRef<double>(&binaryModelResult, 1));
            double cppModelResult = ApplyCatboostModel(floatFeatures, catFeatures);
            bool pass = FuzzyEquals(binaryModelResult, cppModelResult);
            Cout << "BinaryModelResult = " << binaryModelResult << (pass ? " == " : " != ") << "CPPModelResult = " << cppModelResult << '\n';
            return pass;
        };
    };

    Y_UNIT_TEST(CheckOnAdult) {
        TString modelBin = NResource::Find("adult_model_bin");
        TCPPAndBinaryModelsComparator modelsComparator((void*)modelBin.c_str(), modelBin.size());

        /* Lines from adult/test_small */
        UNIT_ASSERT(modelsComparator.CompareOn({39.0, 178100.0, 14.0, 0.0, 0.0, 40.0}, {"0", "n", "1", "Local-gov", "Masters", "Divorced", "Prof-specialty", "Unmarried", "White", "Female", "United-States"}));
        UNIT_ASSERT(modelsComparator.CompareOn({39.0, 178100.0, 14.0, 0.0, 0.0, 40.0}, {"0", "n", "1", "Local-gov", "Masters", "Divorced", "Prof-specialty", "Unmarried", "White", "Female", "United-States"}));
        UNIT_ASSERT(modelsComparator.CompareOn({44.0, 403782.0, 11.0, 0.0, 0.0, 45.0}, {"0", "n", "1", "Private", "Assoc-voc", "Divorced", "Sales", "Not-in-family", "White", "Male", "United-States"}));
        UNIT_ASSERT(modelsComparator.CompareOn({19.0, 208874.0, 10.0, 0.0, 0.0, 40.0}, {"0", "n", "1", "?", "Some-college", "Never-married", "?", "Own-child", "White", "Male", "United-States"}));
        UNIT_ASSERT(modelsComparator.CompareOn({48.0, 236197.0, 8.0, 0.0, 0.0, 40.0}, {"0", "n", "1", "Private", "12th", "Widowed", "Handlers-cleaners", "Not-in-family", "Asian-Pac-Islander", "Male", "Guatemala"}));
        UNIT_ASSERT(modelsComparator.CompareOn({42.0, 121287.0, 9.0, 0.0, 0.0, 45.0}, {"0", "n", "1", "Private", "HS-grad", "Divorced", "Machine-op-inspct", "Not-in-family", "White", "Male", "United-States"}));
    }

    Y_UNIT_TEST(CheckOnUnexpectedInput) {
        TString modelBin = NResource::Find("adult_model_bin");
        TCPPAndBinaryModelsComparator modelsComparator((void*)modelBin.c_str(), modelBin.size());

        UNIT_ASSERT(modelsComparator.CompareOn({0, 0, 0, 0, 0, 0}, {"", "", "", "", "", "", "", "", "", "", ""}));
        std::string longString;
        longString.resize(100000000, '\1');
        UNIT_ASSERT(modelsComparator.CompareOn({-1, -1, 0, 0, 0, 0}, {"abcd", "abcd", "abcd", "\0\0", "", "", "", "", "", "", longString}));
        UNIT_ASSERT(modelsComparator.CompareOn({(float)0.123456789012345, (float)0.123456789012345, 0, 0, 0, 0}, {"0", "n", "1", "improper", "HS-grad", "Divorced", "Divorced", "Divorced", "Divorced", "Divorced", "Divorced"}));
    }
}
