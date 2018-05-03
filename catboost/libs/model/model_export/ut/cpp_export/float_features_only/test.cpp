#include <catboost/libs/model/model.h>

#include <library/unittest/registar.h>
#include <library/unittest/tests_data.h>
#include <library/unittest/env.h>
#include <library/resource/resource.h>

#include <util/generic/ymath.h>

#include <vector>

double ApplyCatboostModel(const std::vector<float>& features);

Y_UNIT_TEST_SUITE(CompareBinaryAndCPPModelNoCatFeatures) {
    class TCPPAndBinaryModelsComparator {
    private:
        TFullModel Calcer;

    public:
        TCPPAndBinaryModelsComparator(const void* buffer, size_t bufferLength) {
            Calcer = ReadModel(buffer, bufferLength);
        };
        bool CompareOn(const std::vector<float>& floatFeatures) {
            TVector<TConstArrayRef<float>> floatFeaturesVec(1);
            TVector<TVector<TStringBuf>> catFeaturesVec(1, TVector<TStringBuf>(0));
            floatFeaturesVec[0] = TConstArrayRef<float>(floatFeatures.data(), floatFeatures.size());

            double binaryModelResult = 0;
            Calcer.Calc(floatFeaturesVec, catFeaturesVec, TArrayRef<double>(&binaryModelResult, 1));
            double cppModelResult = ApplyCatboostModel(floatFeatures);
            bool pass = FuzzyEquals(binaryModelResult, cppModelResult);
            Cout << "BinaryModelResult = " << binaryModelResult << (pass ? " == " : " != ") << "CPPModelResult = " << cppModelResult << '\n';
            return pass;
        };
    };

    Y_UNIT_TEST(CheckOnHiggs) {
        const std::vector<float> test_higgs_line1 = {
            2.089597940444946289e+00, -2.036136239767074585e-01, 1.064085721969604492e+00, 1.543738245964050293e+00, 7.074388861656188965e-01,
            6.418917775154113770e-01, 7.367056608200073242e-01, -1.458505988121032715e+00, 0.000000000000000000e+00, 6.918397545814514160e-01,
            1.209420561790466309e+00, -7.470068931579589844e-01, 2.214872121810913086e+00, 7.270723581314086914e-01, 6.009370684623718262e-01,
            1.264147400856018066e+00, 0.000000000000000000e+00, 1.138171792030334473e+00, -4.443554580211639404e-01, -4.632202535867691040e-02,
            3.101961374282836914e+00, 8.120074272155761719e-01, 1.013427257537841797e+00, 9.982482790946960449e-01, 1.547026276588439941e+00,
            9.381196498870849609e-01, 1.063107132911682129e+00, 1.037893891334533691e+00};
        const std::vector<float> test_higgs_line2 = {
            4.520324766635894775e-01, -2.093112945556640625e+00, -6.239078044891357422e-01, 6.771035790443420410e-01, 5.539864301681518555e-01,
            1.756751656532287598e+00, -1.134828925132751465e+00, 2.455338835716247559e-02, 0.000000000000000000e+00, 2.023772954940795898e+00,
            -5.409948229789733887e-01, -1.519330143928527832e+00, 2.214872121810913086e+00, 8.803898096084594727e-01, 1.315482258796691895e+00,
            -1.547356605529785156e+00, 0.000000000000000000e+00, 1.854032278060913086e+00, 2.194046527147293091e-01, 9.393047094345092773e-01,
            0.000000000000000000e+00, 1.413466334342956543e+00, 1.118542671203613281e+00, 9.920850992202758789e-01, 1.150202035903930664e+00,
            5.419580936431884766e-01, 1.071357965469360352e+00, 1.025450587272644043e+00};
        const std::vector<float> train_higgs_line1 = {
            8.692932128906250000e-01, -6.350818276405334473e-01, 2.256902605295181274e-01, 3.274700641632080078e-01, -6.899932026863098145e-01,
            7.542022466659545898e-01, -2.485731393098831177e-01, -1.092063903808593750e+00, 0.000000000000000000e+00, 1.374992132186889648e+00,
            -6.536741852760314941e-01, 9.303491115570068359e-01, 1.107436060905456543e+00, 1.138904333114624023e+00, -1.578198313713073730e+00,
            -1.046985387802124023e+00, 0.000000000000000000e+00, 6.579295396804809570e-01, -1.045456994324922562e-02, -4.576716944575309753e-02,
            3.101961374282836914e+00, 1.353760004043579102e+00, 9.795631170272827148e-01, 9.780761599540710449e-01, 9.200048446655273438e-01,
            7.216574549674987793e-01, 9.887509346008300781e-01, 8.766783475875854492e-01};
        TString modelBin = NResource::Find("higgs_model_bin");
        TCPPAndBinaryModelsComparator modelsComparator((void*)modelBin.c_str(), modelBin.size());

        UNIT_ASSERT(modelsComparator.CompareOn(test_higgs_line1));
        UNIT_ASSERT(modelsComparator.CompareOn(test_higgs_line2));
        UNIT_ASSERT(modelsComparator.CompareOn(train_higgs_line1));
    }

    Y_UNIT_TEST(CheckOnUnexpectedInput) {
        TString modelBin = NResource::Find("higgs_model_bin");
        TCPPAndBinaryModelsComparator modelsComparator((void*)modelBin.c_str(), modelBin.size());
        UNIT_ASSERT(modelsComparator.CompareOn({NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN}));
        UNIT_ASSERT(modelsComparator.CompareOn({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
        UNIT_ASSERT(modelsComparator.CompareOn({NAN, NAN, NAN, NAN, NAN, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
    }
}
