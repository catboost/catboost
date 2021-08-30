#include "detail.h"

#include <catboost/libs/helpers/exception.h>


EColumn NCB::NQuantizationDetail::IdlColumnTypeToEColumn(NCB::NIdl::EColumnType pbType) {
    EColumn type;
    switch (pbType) {
        case NCB::NIdl::CT_UNKNOWN:
            ythrow TCatBoostException() << "unknown column type in quantized pool";
        case NCB::NIdl::CT_NUMERIC:
            type = EColumn::Num;
            break;
        case NCB::NIdl::CT_LABEL:
            type = EColumn::Label;
            break;
        case NCB::NIdl::CT_WEIGHT:
            type = EColumn::Weight;
            break;
        case NCB::NIdl::CT_GROUP_WEIGHT:
            type = EColumn::GroupWeight;
            break;
        case NCB::NIdl::CT_BASELINE:
            type = EColumn::Baseline;
            break;
        case NCB::NIdl::CT_SUBGROUP_ID:
            type = EColumn::SubgroupId;
            break;
        case NCB::NIdl::CT_DOCUMENT_ID:
            type = EColumn::SampleId;
            break;
        case NCB::NIdl::CT_GROUP_ID:
            type = EColumn::GroupId;
            break;
        case NCB::NIdl::CT_CATEGORICAL:
            type = EColumn::Categ;
            break;
        case NCB::NIdl::CT_SPARSE:
            type = EColumn::Sparse;
            break;
        case NCB::NIdl::CT_TIMESTAMP:
            type = EColumn::Timestamp;
            break;
        case NCB::NIdl::CT_PREDICTION:
            type = EColumn::Prediction;
            break;
        case NCB::NIdl::CT_AUXILIARY:
            type = EColumn::Auxiliary;
            break;
    }
    return type;
}
