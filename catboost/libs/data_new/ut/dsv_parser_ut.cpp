
#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/data_new/dsv_parser.h>
#include <catboost/libs/data_new/meta_info.h>
#include <catboost/libs/data_new/visitor.h>
#include <catboost/libs/helpers/resource_holder.h>
#include <library/unittest/registar.h>

#include <util/generic/maybe.h>
#include <util/generic/variant.h>

using NCB::EDatasetVisitorType;
using NCB::EObjectsOrder;
using NCB::IRawObjectsOrderDataVisitor;
using NCB::IResourceHolder;
using NCB::TDataMetaInfo;
using NCB::TDsvLineParser;
using NCB::TFeaturesLayout;
using NCB::TMaybeData;

template <typename T>
static void SetAtIdx(T value, size_t idx, TVector<T>* arr) {
    if (idx >= arr->size()) {
        arr->resize(idx + 1);
    }

    (*arr)[idx] = std::move(value);
}

namespace {
    struct TTestVisitor : public IRawObjectsOrderDataVisitor {
    public:
        ui32 InBlockObjectIdx = -1;
        bool StartCalled = false;
        bool StartNextBlockCalled = false;
        bool FinishCalled = false;

        TMaybe<TGroupId> GroupId;
        TMaybe<TSubgroupId> SubgroupId;
        TMaybe<ui64> Timestamp;
        TMaybe<TVector<float>> NumericFeatures;
        TMaybe<TVector<ui32>> CategoricalFeatures;
        TVariant<TMonostate, TString, float> Target;
        TMaybe<TVector<float>> Baseline;
        TMaybe<float> Weight;
        TMaybe<float> GroupWeight;

    public:
        explicit TTestVisitor(ui32 inBlockObjectIdx)
            : InBlockObjectIdx(inBlockObjectIdx)
        {
        }

        EDatasetVisitorType GetType() const override {
            return EDatasetVisitorType::RawObjectsOrder;
        }

        void SetGroupWeights(TVector<float>&&) override {
        }

        void SetPairs(TVector<TPair>&&) override {
        }

        TMaybeData<TConstArrayRef<TGroupId>> GetGroupIds() const override {
            return {};
        }

        void Start(
            bool,
            const TDataMetaInfo&,
            ui32,
            EObjectsOrder,
            TVector<TIntrusivePtr<IResourceHolder>>) override
        {
            UNIT_ASSERT_VALUES_EQUAL(StartCalled, false);
            StartCalled = true;
        }

        void StartNextBlock(ui32) override {
            UNIT_ASSERT_VALUES_EQUAL(StartNextBlockCalled, false);
            StartNextBlockCalled = true;
        }

        void AddGroupId(ui32 inBlockObjectIdx, TGroupId value) override {
            UNIT_ASSERT_VALUES_EQUAL(InBlockObjectIdx, inBlockObjectIdx);
            UNIT_ASSERT_VALUES_EQUAL(GroupId.Defined(), false);
            GroupId = value;
        }

        void AddSubgroupId(ui32 inBlockObjectIdx, TSubgroupId value) override {
            UNIT_ASSERT_VALUES_EQUAL(InBlockObjectIdx, inBlockObjectIdx);
            UNIT_ASSERT_VALUES_EQUAL(SubgroupId.Defined(), false);
            SubgroupId = value;
        }

        void AddTimestamp(ui32 inBlockObjectIdx, ui64 value) override {
            UNIT_ASSERT_VALUES_EQUAL(InBlockObjectIdx, inBlockObjectIdx);
            UNIT_ASSERT_VALUES_EQUAL(Timestamp.Defined(), false);
            Timestamp = value;
        }

        void AddFloatFeature(ui32 inBlockObjectIdx, ui32 catFeatureIdx, float value) override {
            UNIT_ASSERT_VALUES_EQUAL(InBlockObjectIdx, inBlockObjectIdx);
            if (!NumericFeatures.Defined()) {
                NumericFeatures.ConstructInPlace();
            }

            SetAtIdx(value, catFeatureIdx, &NumericFeatures.GetRef());
        }

        void AddAllFloatFeatures(ui32 inBlockObjectIdx, TConstArrayRef<float> values) override {
            UNIT_ASSERT_VALUES_EQUAL(InBlockObjectIdx, inBlockObjectIdx);
            UNIT_ASSERT_VALUES_EQUAL(NumericFeatures.Defined(), false);
            NumericFeatures.ConstructInPlace();
            NumericFeatures->assign(values.begin(), values.end());
        }

        ui32 GetCatFeatureValue(ui32 catFeatureIdx, TStringBuf value) override {
            (void)catFeatureIdx;
            return CalcCatFeatureHash(value);
        }

        void AddCatFeature(ui32 inBlockObjectIdx, ui32 catFeatureIdx, TStringBuf value) override {
            UNIT_ASSERT_VALUES_EQUAL(InBlockObjectIdx, inBlockObjectIdx);
            if (!CategoricalFeatures.Defined()) {
                CategoricalFeatures.ConstructInPlace();
            }

            SetAtIdx(GetCatFeatureValue(catFeatureIdx, value), catFeatureIdx, &CategoricalFeatures.GetRef());
        }

        void AddAllCatFeatures(ui32 inBlockObjectIdx, TConstArrayRef<ui32> values) override {
            UNIT_ASSERT_VALUES_EQUAL(InBlockObjectIdx, inBlockObjectIdx);
            UNIT_ASSERT_VALUES_EQUAL(CategoricalFeatures.Defined(), false);
            CategoricalFeatures.ConstructInPlace();
            CategoricalFeatures->assign(values.begin(), values.end());
        }

        void AddTarget(ui32 inBlockObjectIdx, const TString& value) override {
            UNIT_ASSERT_VALUES_EQUAL(InBlockObjectIdx, inBlockObjectIdx);
            UNIT_ASSERT_VALUES_EQUAL(static_cast<bool>(GetIf<TMonostate>(&Target)), true);
            Target = value;
        }

        void AddTarget(ui32 inBlockObjectIdx, float value) override {
            UNIT_ASSERT_VALUES_EQUAL(InBlockObjectIdx, inBlockObjectIdx);
            UNIT_ASSERT_VALUES_EQUAL(static_cast<bool>(GetIf<TMonostate>(&Target)), true);
            Target = value;
        }

        void AddBaseline(ui32 inBlockObjectIdx, ui32 baselineIdx, float value) override {
            UNIT_ASSERT_VALUES_EQUAL(InBlockObjectIdx, inBlockObjectIdx);
            if (!Baseline.Defined()) {
                Baseline.ConstructInPlace();
            }
            SetAtIdx(value, baselineIdx, &Baseline.GetRef());
        }

        void AddWeight(ui32 inBlockObjectIdx, float value) override {
            UNIT_ASSERT_VALUES_EQUAL(InBlockObjectIdx, inBlockObjectIdx);
            UNIT_ASSERT_VALUES_EQUAL(Weight.Defined(), false);
            Weight = value;
        }

        void AddGroupWeight(ui32 inBlockObjectIdx, float value) override {
            UNIT_ASSERT_VALUES_EQUAL(InBlockObjectIdx, inBlockObjectIdx);
            UNIT_ASSERT_VALUES_EQUAL(GroupWeight.Defined(), false);
            GroupWeight = value;
        }

        void Finish() override {
            UNIT_ASSERT_VALUES_EQUAL(FinishCalled, false);
            FinishCalled = true;
        }
    };
}

Y_UNIT_TEST_SUITE(DsvLineParserTests) {
    Y_UNIT_TEST(Test1) {
        const char line[] = "0 1";
        const ui32 lineIdx = 0;
        const char delimiter = ' ';
        const TColumn columnsDescription[] = {
            TColumn{EColumn::Label, ""},
            TColumn{EColumn::Num, ""}};
        const bool featureIgnored[] = {false, false};
        const TFeaturesLayout layout(1);
        float numericFeaturesBuffer[1];
        TTestVisitor visitor(lineIdx);
        TDsvLineParser parser(
            delimiter,
            columnsDescription,
            featureIgnored,
            &layout,
            numericFeaturesBuffer,
            {},
            &visitor);

        const auto err = parser.Parse(line, lineIdx);

        UNIT_ASSERT_VALUES_EQUAL(false, static_cast<bool>(err));
        UNIT_ASSERT_VALUES_EQUAL(false, visitor.StartCalled);
        UNIT_ASSERT_VALUES_EQUAL(false, visitor.StartNextBlockCalled);
        UNIT_ASSERT_VALUES_EQUAL(false, visitor.FinishCalled);

        UNIT_ASSERT_VALUES_EQUAL(false, visitor.GroupId.Defined());
        UNIT_ASSERT_VALUES_EQUAL(false, visitor.SubgroupId.Defined());
        UNIT_ASSERT_VALUES_EQUAL(false, visitor.Timestamp.Defined());
        UNIT_ASSERT_VALUES_EQUAL(true, visitor.NumericFeatures.Defined());
        UNIT_ASSERT_VALUES_EQUAL(false, visitor.CategoricalFeatures.Defined());
        UNIT_ASSERT_VALUES_EQUAL(false, static_cast<bool>(GetIf<TMonostate>(&visitor.Target)));
        UNIT_ASSERT_VALUES_EQUAL(false, visitor.Baseline.Defined());
        UNIT_ASSERT_VALUES_EQUAL(false, visitor.Weight.Defined());
        UNIT_ASSERT_VALUES_EQUAL(false, visitor.GroupWeight.Defined());

        UNIT_ASSERT_VALUES_EQUAL(true, static_cast<bool>(GetIf<TString>(&visitor.Target)));
        UNIT_ASSERT_VALUES_EQUAL("0", Get<TString>(visitor.Target));

        UNIT_ASSERT_VALUES_EQUAL(1, visitor.NumericFeatures.GetRef().size());
        UNIT_ASSERT_VALUES_EQUAL(TVector<float>{1.f}, visitor.NumericFeatures.GetRef());
    }

    Y_UNIT_TEST(Test2) {
        const char line[] = "0 1 2 3 4 5 6 7 8 9 10 11 12";
        const ui32 lineIdx = 0;
        const char delimiter = ' ';
        const TColumn columnsDescription[] = {
            TColumn{EColumn::Label, ""},  // 0
            TColumn{EColumn::Num, ""},  // 1
            TColumn{EColumn::Categ, ""},  // 2
            TColumn{EColumn::Num, ""},  // 3
            TColumn{EColumn::Categ, ""},  // 4
            TColumn{EColumn::Auxiliary, ""},  // 5
            TColumn{EColumn::Baseline, ""},  // 6
            TColumn{EColumn::Weight, ""},  // 7
            TColumn{EColumn::GroupId, ""},  // 8
            TColumn{EColumn::GroupWeight, ""},  // 9
            TColumn{EColumn::SubgroupId, ""},  // 10
            TColumn{EColumn::Timestamp, ""},  // 11
            TColumn{EColumn::Baseline, ""}};  // 12
        const bool featureIgnored[] = {false, false, false, false};
        const TFeaturesLayout layout(4, {1, 3}, {"", "", "", ""}, nullptr);
        float numericFeaturesBuffer[2];
        ui32 categoricalFeaturesBuffer[2];
        TTestVisitor visitor(lineIdx);
        TDsvLineParser parser(
            delimiter,
            columnsDescription,
            featureIgnored,
            &layout,
            numericFeaturesBuffer,
            categoricalFeaturesBuffer,
            &visitor);

        const auto err = parser.Parse(line, lineIdx);

        UNIT_ASSERT_VALUES_EQUAL(false, static_cast<bool>(err));
        UNIT_ASSERT_VALUES_EQUAL(false, visitor.StartCalled);
        UNIT_ASSERT_VALUES_EQUAL(false, visitor.StartNextBlockCalled);
        UNIT_ASSERT_VALUES_EQUAL(false, visitor.FinishCalled);

        UNIT_ASSERT_VALUES_EQUAL(true, visitor.GroupId.Defined());
        UNIT_ASSERT_VALUES_EQUAL(true, visitor.SubgroupId.Defined());
        UNIT_ASSERT_VALUES_EQUAL(true, visitor.Timestamp.Defined());
        UNIT_ASSERT_VALUES_EQUAL(true, visitor.NumericFeatures.Defined());
        UNIT_ASSERT_VALUES_EQUAL(true, visitor.CategoricalFeatures.Defined());
        UNIT_ASSERT_VALUES_EQUAL(false, static_cast<bool>(GetIf<TMonostate>(&visitor.Target)));
        UNIT_ASSERT_VALUES_EQUAL(true, visitor.Baseline.Defined());
        UNIT_ASSERT_VALUES_EQUAL(true, visitor.Weight.Defined());
        UNIT_ASSERT_VALUES_EQUAL(true, visitor.GroupWeight.Defined());

        UNIT_ASSERT_VALUES_EQUAL(true, static_cast<bool>(GetIf<TString>(&visitor.Target)));
        UNIT_ASSERT_VALUES_EQUAL("0", Get<TString>(visitor.Target));

        UNIT_ASSERT_VALUES_EQUAL(2, visitor.Baseline.GetRef().size());
        UNIT_ASSERT_VALUES_EQUAL((TVector<float>{6.f, 12.f}), visitor.Baseline.GetRef());

        UNIT_ASSERT_VALUES_EQUAL(7, visitor.Weight.GetRef());
        UNIT_ASSERT_VALUES_EQUAL(CalcGroupIdFor("8"), visitor.GroupId.GetRef());
        UNIT_ASSERT_VALUES_EQUAL(9, visitor.GroupWeight.GetRef());
        UNIT_ASSERT_VALUES_EQUAL(CalcSubgroupIdFor("10"), visitor.SubgroupId.GetRef());
        UNIT_ASSERT_VALUES_EQUAL(11, visitor.Timestamp.GetRef());

        UNIT_ASSERT_VALUES_EQUAL(2, visitor.NumericFeatures.GetRef().size());
        UNIT_ASSERT_VALUES_EQUAL((TVector<float>{1.f, 3.f}), visitor.NumericFeatures.GetRef());

        UNIT_ASSERT_VALUES_EQUAL(2, visitor.CategoricalFeatures.GetRef().size());
        const TVector<ui32> catFeatureHashesExpected = {
            CalcCatFeatureHash("2"),
            CalcCatFeatureHash("4")};
        UNIT_ASSERT_VALUES_EQUAL(catFeatureHashesExpected, visitor.CategoricalFeatures.GetRef());
    }
}
