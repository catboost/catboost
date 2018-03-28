#include <catboost/libs/data/load_data.h>

#include <library/threading/local_executor/local_executor.h>

#include <library/unittest/registar.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/random/fast.h>
#include <util/generic/guid.h>
#include <util/stream/file.h>

using namespace std;

SIMPLE_UNIT_TEST_SUITE(TDataLoadTest) {
    //
    SIMPLE_UNIT_TEST(TestFileRead) {
        TReallyFastRng32 rng(1);
        const size_t TestDocCount = 20000;
        const size_t FactorCount = 250;
        TDocumentStorage documents;
        documents.Resize(TestDocCount, FactorCount, /*baseline dimension*/ 0, /*hasQueryId*/ false, /*hasSubgroupId*/ false);
        for (size_t i = 0; i < TestDocCount; ++i) {
            documents.Target[i] = rng.GenRandReal2();
            for (size_t j = 0; j < FactorCount; ++j) {
                documents.Factors[j][i] = rng.GenRandReal2();
            }
        }
        TString TestFileName = "sample_pool.tsv";
        {
            TOFStream writer(TestFileName);
            for (size_t docIdx = 0; docIdx < documents.GetDocCount(); ++docIdx) {
                writer << documents.Target[docIdx];
                for (const auto& factor : documents.Factors) {
                    writer << "\t" << factor[docIdx];
                }
                writer << Endl;
            }
        }
        TPool pool;
        ReadPool("", TestFileName, "", /*ignoredFeatures*/ {}, 2, false, '\t', false, TVector<TString>(), &pool);
        UNIT_ASSERT_EQUAL(pool.Docs.GetDocCount(), documents.GetDocCount());
        UNIT_ASSERT_EQUAL(pool.Docs.GetFactorsCount(), documents.GetFactorsCount());
        for (int j = 0; j < documents.GetFactorsCount(); ++j) {
            const auto& redFactors = pool.Docs.Factors[j];
            const auto& factors = documents.Factors[j];
            for (size_t i = 0; i < documents.GetDocCount(); ++i) {
                UNIT_ASSERT_DOUBLES_EQUAL(pool.Docs.Target[i], documents.Target[i], 1e-5);
                UNIT_ASSERT_DOUBLES_EQUAL(factors[i], redFactors[i], 1e-5);
            }
        }
    }
}
