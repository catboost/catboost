#pragma once

#include <catboost/cuda/data/data_provider.h>
#include <catboost/cuda/data/grid_creator.h>
#include <catboost/cuda/data/proto_helpers.h>

#include <catboost/cuda/cuda_lib/helpers.h>
#include <catboost/libs/data/load_helpers.h>
#include <catboost/cuda/data/pool_proto/pool.pb.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/column_description/cd_parser.h>

#include <library/protobuf/protofile/protofile.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/stream/file.h>
#include <util/generic/vector.h>
#include <util/generic/map.h>
#include <util/system/fs.h>
#include <util/string/builder.h>
#include <util/system/tempfile.h>
#include <util/string/split.h>
#include <util/system/mktemp.h>
#include <catboost/libs/options/binarization_options.h>

namespace NCatboostCuda {
    template <class T, template <class U> class TField>
    inline void VectorToProto(const TVector<T>& src, TField<T>* dst) {
        *dst = TField<T>(src.begin(), src.end());
    }

    inline void SetBinarizedData(NCompressedPool::TFeatureColumn& column, const TVector<ui64>& data, ui32 bitsPerKey, ui32 len) {
        VectorToProto(data, column.MutableBinarizedColumn()->MutableData());
        column.MutableBinarizedColumn()->SetBitsPerKey(bitsPerKey);
        column.MutableBinarizedColumn()->SetLength(len);
    }

    template <class TColumn>
    inline void SetBorders(TColumn& column, const TVector<float>& borders) {
        VectorToProto(borders, column.MutableBinarization()->MutableBorders());
    }

    class TSplittedByColumnsTempPool {
    public:
        TSplittedByColumnsTempPool(const TString& tempDir,
                                   const TString& pool,
                                   const TVector<TColumn>& columnDescription)
            : TempDir(tempDir)
            , ColumnsDescription(columnDescription)
        {
            NFs::MakeDirectory(tempDir);
            Split(pool);
        }

        void ReadColumn(ui32 column,
                        TVector<TString>& dst) {
            CB_ENSURE(SplitDone);
            CB_ENSURE(column < Columns.size());
            CB_ENSURE(Columns[column]);
            dst.resize(LineCount);

            TIFStream input(Columns[column]->Name());

            for (ui32 line = 0; line < LineCount; ++line) {
                ::Load(&input, dst[line]);
            }
        }

        ui32 GetLineCount() const {
            return LineCount;
        }

    private:
        void Split(const TString& pool) {
            ui32 columnCount = ColumnsDescription.size();
            MATRIXNET_DEBUG_LOG << "Column count " << columnCount << Endl;
            Y_ENSURE(columnCount >= (ui32)ReadColumnsCount(pool), "Error: found too many columns in cd-file");
            Columns.resize(0);
            Columns.resize(columnCount);
            TVector<THolder<TOFStream>> outputs;
            outputs.resize(columnCount);

            for (ui32 i = 0; i < Columns.size(); ++i) {
                if (ColumnsDescription[i].Type == EColumn::Auxiliary) {
                    continue;
                }
                Columns[i] = MakeHolder<TTempFile>(TStringBuilder() << TempDir.data() << "/" << i << ".column");
                outputs[i] = MakeHolder<TOFStream>(Columns[i]->Name());
            }

            TIFStream input(pool);
            TVector<TStringBuf> words;
            TString line;

            while (input.ReadLine(line)) {
                words.clear();
                SplitRangeTo<const char, TVector<TStringBuf>>(~line, ~line + line.size(), '\t', &words);
                CB_ENSURE(words.ysize() == ColumnsDescription.ysize(),
                          TStringBuilder() << "Wrong columns number in pool line: Found " << words.size()
                                           << "; Expected " << ColumnsDescription.size());
                for (int i = 0; i < words.ysize(); ++i) {
                    switch (ColumnsDescription[i].Type) {
                        case EColumn::Auxiliary: {
                            break;
                        }
                        default: {
                            CB_ENSURE(words[i].ToString().Size());
                            const TString word = words[i].ToString();
                            ::Save(outputs[i].Get(), word);
                            break;
                        }
                    }
                }
                ++LineCount;
            }
            SplitDone = true;
        }

        TVector<THolder<TTempFile>> Columns;

        TString TempDir;
        const TVector<TColumn>& ColumnsDescription;

        bool SplitDone = false;
        ui32 LineCount = 0;
    };

    class TColumnConverter {
    public:
        TColumnConverter(ui32 factorId, ui32 columnId, TString factorName)
            : FactorId(factorId)
            , ColumnId(columnId)
            , FactorName(factorName)
        {
        }

        virtual ~TColumnConverter() {
        }

    protected:
        ui32 FactorId = 0;
        ui32 ColumnId = 0;
        TString FactorName = "";

        void WriteCommonDescription(::NCompressedPool::TFeatureDescription& description) {
            description.SetFeatureId(FactorId);
            description.SetColumnId(ColumnId);
            description.SetFeatureName(FactorName);
        }
    };

    class TFloatColumnConverter: public TColumnConverter {
    public:
        TFloatColumnConverter(ui32 factorId, ui32 columnId, const TString& factorName)
            : TColumnConverter(factorId,
                               columnId,
                               factorName) {
        }

        TFloatColumnConverter& SetColumn(const TVector<TString>& column) {
            FloatColumn.resize(column.size());

            NPar::ParallelFor(0, (ui32)column.size(), [&](int i) {
                float val = 0;
                CB_ENSURE(column[i] != "nan", "nan values not supported");
                CB_ENSURE(column[i] != "", "empty values not supported");
                CB_ENSURE(TryFromString<float>(column[i], val),
                          "Can not parse float factor value " << column[i] << " in column " << i
                                                              << ". Try correcting column description file.");
                FloatColumn[i] = val;
            });

            return *this;
        }

        TFloatColumnConverter& BuildBinarized(IFactory<IGridBuilder>& gridBuilderFactory,
                                              const NCatboostOptions::TBinarizationOptions& config) {
            CB_ENSURE(FloatColumn.size(), "Set float column first");
            auto borders = TBordersBuilder(gridBuilderFactory, FloatColumn)(config);
            return BuildBinarized(std::move(borders));
        }

        void Write(NCompressedPool::TFeatureColumn& column) {
            column.Clear();
            auto* description = column.MutableFeatureDescription();
            WriteCommonDescription(*description);
            if (BinarizeIt) {
                auto binarizedFeature = BinarizeLine(~FloatColumn, FloatColumn.size(), ENanMode::Forbidden, Borders);
                const auto bitsPerKey = IntLog2(Borders.size() + 1);
                TVector<ui64> compressedLine;
                if (bitsPerKey) {
                    compressedLine = CompressVector<ui64>(binarizedFeature, bitsPerKey);
                }
                MATRIXNET_INFO_LOG << "Compressed feature " << description->GetFeatureId() << " from "
                                   << sizeof(float) * FloatColumn.size() / 1024 / 1024 << " to "
                                   << sizeof(ui64) * compressedLine.size() / 1024 / 1024 << Endl;
                MATRIXNET_INFO_LOG << "Bits per key " << bitsPerKey << Endl;
                SetBinarizedData(column, compressedLine, bitsPerKey, FloatColumn.size());
                SetBorders(column, Borders);
                description->SetFeatureType(::NCompressedPool::TFeatureType::Binarized);
            } else {
                *column.MutableFloatColumn()->MutableValues() = ::google::protobuf::RepeatedField<float>(
                    FloatColumn.begin(), FloatColumn.end());
                description->SetFeatureType(::NCompressedPool::TFeatureType::Float);
            }
        }

        TFloatColumnConverter& Clear() {
            FloatColumn.clear();
            Borders.clear();
            BinarizeIt = false;
            return *this;
        }

        void WriteBinarization(NCompressedPool::TFloatFeatureBinarization& binarization) {
            Y_ENSURE(BinarizeIt);
            const auto description = binarization.MutableFeatureDescription();
            WriteCommonDescription(*description);
            SetBorders(binarization, Borders);
            description->SetFeatureType(::NCompressedPool::TFeatureType::Binarized);
        }

        TFloatColumnConverter& BuildBinarized(TVector<float>&& borders) {
            Borders = std::move(borders);
            BinarizeIt = true;
            return *this;
        }

    private:
        TVector<float> FloatColumn;
        TVector<float> Borders;
        bool BinarizeIt = false;
    };

    class TCatFeatureColumnConverter: public TColumnConverter {
    public:
        TCatFeatureColumnConverter(ui32 factorId, ui32 columnId, const TString& factorName)
            : TColumnConverter(factorId,
                               columnId,
                               factorName) {
        }

        TCatFeatureColumnConverter& AddExistingBinarization(TMap<TString, ui32>&& binarization) {
            Binarization = std::move(binarization);
            return *this;
        }

        TCatFeatureColumnConverter& AddKeys(const TVector<TString>& column) {
            for (const auto& entry : column) {
                if (!Binarization.has(entry)) {
                    Binarization[entry] = Binarization.size();
                }
            }
            return *this;
        }

        TCatFeatureColumnConverter& SetColumn(const TVector<TString>& column) {
            BinarizedData.resize_uninitialized(column.size());
            NPar::ParallelFor(0, (ui32)column.size(), [&](int i) {
                const auto& key = column[i];
                BinarizedData[i] = Binarization.at(key);
            });

            return *this;
        }

        TCatFeatureColumnConverter Write(NCompressedPool::TFeatureColumn& column) {
            column.Clear();
            WriteColumnDescription(column);

            const ui32 uniqueValues = (const ui32)Binarization.size();
            const auto bitsPerKey = IntLog2(uniqueValues);
            auto compressedVector = CompressVector<ui64>(BinarizedData, bitsPerKey);
            MATRIXNET_INFO_LOG << "Compressed cat feature " << column.GetFeatureDescription().GetFeatureId() << " from "
                               << sizeof(ui32) * BinarizedData.size() / 1024 / 1024 << " to "
                               << sizeof(ui64) * compressedVector.size() / 1024 / 1024 << Endl;
            MATRIXNET_INFO_LOG << "Bits per key " << bitsPerKey << Endl;
            SetBinarizedData(column, compressedVector, bitsPerKey, BinarizedData.size());

            column.SetUniqueValues(uniqueValues);
            return *this;
        }

        template <class TColumn>
        void WriteColumnDescription(TColumn& column) {
            auto* description = column.MutableFeatureDescription();
            //maybe refactor it, cat feature type should be set
            WriteCommonDescription(*description);
            description->SetFeatureType(NCompressedPool::Categorical);
        }

        TCatFeatureColumnConverter& WriteBinarization(NCompressedPool::TCatFeatureBinarization& binarization) {
            binarization.Clear();
            WriteColumnDescription(binarization);

            TVector<TString> keys;
            TVector<ui32> bins;

            for (auto& entry : Binarization) {
                keys.push_back(entry.first);
                bins.push_back(entry.second);
            }

            VectorToProto(keys, binarization.MutableKeys());
            VectorToProto(bins, binarization.MutableBins());

            return *this;
        }

    private:
        TMap<TString, ui32> Binarization;
        TVector<ui32> BinarizedData;
    };

    class TCatBoostProtoPoolConverter {
    public:
        TCatBoostProtoPoolConverter(const TString& pool,
                                    const TString& poolCd,
                                    TString tempDir = "tmp")
            : ColumnsDescription(ReadCD(poolCd,
                                        TCdParserDefaults(EColumn::Num, ReadColumnsCount(pool))))
            , SplittedPool(tempDir, pool, ColumnsDescription)
        {
        }

        TCatBoostProtoPoolConverter& SetGridBuilderFactory(IFactory<IGridBuilder>& builderFactory) {
            GridBuilderFactory = &builderFactory;
            return *this;
        }

        TCatBoostProtoPoolConverter& SetBinarization(const NCatboostOptions::TBinarizationOptions& binarizationConfig) {
            BinarizationConfiguration = &binarizationConfig;
            return *this;
        }

        TCatBoostProtoPoolConverter& SetOutputFile(const TString& file) {
            Output.Reset(new TOFStream(file));
            return *this;
        }

        TCatBoostProtoPoolConverter& SetOutputBinarizationFile(const TString& file) {
            OutputBinarization.Reset(new TOFStream(file));
            return *this;
        }

        TCatBoostProtoPoolConverter& SetInputBinarizationFile(const TString& file) {
            Y_ENSURE(BinarizationConfiguration == nullptr,
                     "Error: can't use binarization configuration and existing proto-binarization at the same time");
            InputBinarization.Reset(new TIFStream(file));
            return *this;
        }

        void Convert() {
            CB_ENSURE(Output, "Please provide output file");

            int targetColumn = -1;
            int docIdColumn = -1;
            int groupIdColumn = -1;
            int subgroupIdColumn = -1;
            int weightColumn = -1;
            int timestampColumn = -1;

            TVector<ui32> baselineColumns;

            TVector<ui32> featureColumns;

            for (ui32 col = 0; col < ColumnsDescription.size(); ++col) {
                const auto type = ColumnsDescription[col].Type;
                switch (type) {
                    case EColumn::Categ:
                    case EColumn::Num: {
                        featureColumns.push_back(col);
                        break;
                    }
                    case EColumn::Label: {
                        CB_ENSURE(targetColumn == -1, "Error: more than one Label column");

                        targetColumn = col;
                        break;
                    }
                    case EColumn::Weight: {
                        CB_ENSURE(weightColumn == -1, "Error: more than one Weight column");
                        weightColumn = col;
                        break;
                    }
                    case EColumn::DocId: {
                        CB_ENSURE(docIdColumn == -1, "Error: more than one DocId column");
                        docIdColumn = col;
                        break;
                    }
                    case EColumn::GroupId: {
                        CB_ENSURE(groupIdColumn == -1, "Error: more than one GroupId column.");
                        groupIdColumn = col;
                        break;
                    }
                    case EColumn::SubgroupId: {
                        CB_ENSURE(subgroupIdColumn == -1, "Error: more than one GroupId column.");
                        subgroupIdColumn = col;
                        break;
                    }
                    case EColumn::Baseline: {
                        baselineColumns.push_back(col);
                        break;
                    }
                    case EColumn::Timestamp: {
                        CB_ENSURE(timestampColumn == -1, "Error: more than one timestamp column.");
                        timestampColumn = col;
                        break;
                    }
                    case EColumn::Auxiliary: {
                        break;
                    }
                    default: {
                        ythrow TCatboostException() << "Error: unknown column type " << type;
                    }
                }
            }

            NCompressedPool::TFloatColumn floatColumn;

            //header
            {
                NCompressedPool::TPoolStructure poolStructure;
                poolStructure.SetDocCount(SplittedPool.GetLineCount());
                poolStructure.SetFeatureCount((google::protobuf::uint32)featureColumns.size());

                poolStructure.SetBaselineColumn(baselineColumns.size());
                poolStructure.SetDocIdColumn(docIdColumn != -1);
                poolStructure.SetTimestampColumn(timestampColumn != -1);
                poolStructure.SetGroupIdColumn(groupIdColumn != -1);
                poolStructure.SetSubgroupIdColumn(subgroupIdColumn != -1);
                poolStructure.SetWeightColumn(weightColumn != -1);
                WriteMessage(poolStructure, Output.Get());
            }
            //targets/qids/other meta columns

            //target
            {
                ConvertColumn<float, NCompressedPool::TFloatColumn>(targetColumn, floatColumn);
                WriteMessage(floatColumn, Output.Get());
            }

            if (weightColumn != -1) {
                ConvertColumn<float, NCompressedPool::TFloatColumn>(weightColumn, floatColumn);
                WriteMessage(floatColumn, Output.Get());
            }

            //docId, queryId
            {
                NCompressedPool::TUnsignedIntegerColumn uintColumn;
                if (docIdColumn != -1) {
                    ConvertColumn<ui32, NCompressedPool::TUnsignedIntegerColumn>(docIdColumn, uintColumn);
                }
                if (timestampColumn != -1) {
                    ConvertColumn<ui64, NCompressedPool::TUnsignedIntegerColumn>(timestampColumn, uintColumn);
                }

                if (groupIdColumn != -1) {
                    NCompressedPool::TIntegerColumn intColumn;
                    ConvertColumnAndWrite<ui32, NCompressedPool::TIntegerColumn>(groupIdColumn, intColumn,
                                                                                 [](const TString& str,
                                                                                    ui32& val) -> bool {
                                                                                     val = StringToIntHash(str);
                                                                                     return true;
                                                                                 });
                }
                if (subgroupIdColumn != -1) {
                    NCompressedPool::TIntegerColumn intColumn;
                    ConvertColumnAndWrite<ui32, NCompressedPool::TIntegerColumn>(groupIdColumn, intColumn,
                                                                                 [](const TString& str,
                                                                                    ui32& val) -> bool {
                                                                                     val = StringToIntHash(str);
                                                                                     return true;
                                                                                 });
                }
            }

            for (ui32 baselineColumn : baselineColumns) {
                ConvertColumn<float, NCompressedPool::TFloatColumn>(baselineColumn, floatColumn);
                WriteMessage(floatColumn, Output.Get());
            }

            if (!BinarizationConfiguration) {
                MATRIXNET_WARNING_LOG << "Converting pool without float features binarizatin" << Endl;
            }

            //features
            {
                NCompressedPool::TFeatureColumn protoFeatureColumn;
                TVector<TString> factorColumn;
                NCompressedPool::TCatFeatureBinarization binarization;
                NCompressedPool::TFloatFeatureBinarization floatFeatureBinarization;

                for (ui32 i = 0; i < featureColumns.size(); ++i) {
                    protoFeatureColumn.Clear();
                    const ui32 columnId = featureColumns[i];
                    const auto description = ColumnsDescription[featureColumns[i]];
                    SplittedPool.ReadColumn(columnId, factorColumn);

                    switch (description.Type) {
                        //for float features we first convert, than build grid
                        case EColumn::Num: {
                            TFloatColumnConverter converter(i,
                                                            columnId,
                                                            description.Id);
                            converter.SetColumn(factorColumn);

                            if (BinarizationConfiguration) {
                                if (!InputBinarization) {
                                    Y_ENSURE(GridBuilderFactory);
                                    converter.BuildBinarized(*GridBuilderFactory,
                                                             *BinarizationConfiguration);
                                } else {
                                    ReadMessage(*InputBinarization, floatFeatureBinarization);
                                    Y_ENSURE(floatFeatureBinarization.GetFeatureDescription().GetFeatureId() == i,
                                             "Error: featureId in index should be equal to featureId in converted pool");
                                    TVector<float> borders(floatFeatureBinarization.GetBinarization().borders().begin(),
                                                           floatFeatureBinarization.GetBinarization().borders().end());
                                    converter.BuildBinarized(std::move(borders));
                                }
                                if (OutputBinarization) {
                                    converter.WriteBinarization(floatFeatureBinarization);
                                    WriteMessage(floatFeatureBinarization, OutputBinarization.Get());
                                }
                            }
                            converter.Write(protoFeatureColumn);

                            break;
                        }
                            //warning: for cat features we first build binarization, and only than convert
                        case EColumn::Categ: {
                            TCatFeatureColumnConverter converter(i, columnId, description.Id);
                            if (InputBinarization) {
                                ReadMessage(*InputBinarization, binarization);
                                TMap<TString, ui32> bins;
                                for (int i = 0; i < binarization.GetBins().size(); ++i) {
                                    bins[binarization.GetKeys(i)] = binarization.GetBins().Get(i);
                                }

                                Y_ENSURE(binarization.GetFeatureDescription().GetFeatureId() == i,
                                         "Error: featureId in index should be equal to featureId in converted pool");
                                converter.AddExistingBinarization(std::move(bins));
                            }
                            //build index
                            converter.AddKeys(factorColumn);
                            if (OutputBinarization) {
                                converter.WriteBinarization(binarization);
                                WriteMessage(binarization, OutputBinarization.Get());
                            }
                            //binarize
                            converter.SetColumn(factorColumn);
                            converter.Write(protoFeatureColumn);
                            break;
                        }
                        default: {
                            ythrow TCatboostException() << "Unsuppored feature type";
                        }
                    }
                    WriteMessage(protoFeatureColumn, Output.Get());
                }
            }
        }

    private:
        template <class T, class TColumn, class TFunction>
        inline void ConvertColumnAndWrite(ui32 columnId,
                                          TColumn& column,
                                          TFunction&& cast) {
            TVector<TString> strColumn;
            SplittedPool.ReadColumn(columnId, strColumn);
            column.MutableValues()->Resize(strColumn.size(), 0);
            NPar::ParallelFor(0, strColumn.size(), [&](int i) -> void {
                T val;
                CB_ENSURE(cast(strColumn[i], val), "Column #" + ToString(columnId) +
                                                       " cannot be parsed. Try correcting column description file. Failed on line " +
                                                       ToString(i));
                column.MutableValues()->Set(i, val);
            });
            WriteMessage(column, Output.Get());
        }

        template <class T, class TColumn>
        inline void ConvertColumn(ui32 columnId,
                                  TColumn& column) {
            ConvertColumnAndWrite<T, TColumn>(columnId, column, [](const TString& str, T& val) -> bool {
                return TryFromString<T>(str, val);
            });
        };

        template <class TProtoMessage>
        inline void WriteMessage(TProtoMessage& message, TOFStream* output) {
            NFastTier::TBinaryProtoWriter<TProtoMessage> messageWriter;
            messageWriter.Open(output);
            messageWriter.Write(message);
        }

    private:
        THolder<TOFStream> Output;
        THolder<TOFStream> OutputBinarization;

        THolder<TIFStream> InputBinarization;

        TVector<TColumn> ColumnsDescription;
        TSplittedByColumnsTempPool SplittedPool;

        const NCatboostOptions::TBinarizationOptions* BinarizationConfiguration;
        IFactory<IGridBuilder>* GridBuilderFactory = nullptr;
    };
}
