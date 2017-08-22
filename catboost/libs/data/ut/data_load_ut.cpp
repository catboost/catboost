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
        yvector<TDocInfo> documents;
        for (size_t i = 0; i < TestDocCount; ++i) {
            TDocInfo doc;
            doc.Target = (float)rng.GenRandReal2();
            doc.Factors.resize(FactorCount);
            for (size_t j = 0; j < FactorCount; ++j) {
                doc.Factors[j] = (float)rng.GenRandReal2();
            }
            documents.emplace_back(std::move(doc));
        }
        TString TestFileName = "sample_pool.tsv";
        {
            TOFStream writer(TestFileName);
            for (const auto& doc : documents) {
                writer << doc.Target;
                for (const auto& factor : doc.Factors) {
                    writer << "\t" << factor;
                }
                writer << Endl;
            }
        }
        TPool pool;
        ReadPool("", TestFileName, 2, false, '\t', false, yvector<TString>(), &pool);
        yvector<TDocInfo>& readTestDocuments = pool.Docs;
        UNIT_ASSERT_EQUAL(readTestDocuments.size(), documents.size());
        for (size_t i = 0; i < documents.size(); ++i) {
            const auto& doc1 = documents[i];
            const auto& doc2 = readTestDocuments[i];
            UNIT_ASSERT_DOUBLES_EQUAL(doc1.Target, doc2.Target, 1e-5);
            UNIT_ASSERT_EQUAL(doc1.Factors.size(), doc2.Factors.size());
            for (size_t j = 0; j < doc1.Factors.size(); ++j) {
                UNIT_ASSERT_DOUBLES_EQUAL(doc1.Factors[j], doc2.Factors[j], 1e-5);
            }
        }
    }
}
