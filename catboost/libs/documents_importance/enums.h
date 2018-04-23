#pragma once

enum class EDocumentStrengthType {
    PerObject,
    Average,
    Raw
};

enum class EUpdateType {
    SinglePoint,
    AllPoints,
    TopKLeaves
};

enum class EImportanceValuesSign {
    Positive,
    Negative,
    All
};
