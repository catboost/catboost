#pragma once

#include <catboost/libs/data_new/data_provider.h>

#include <util/generic/fwd.h>

namespace NCB {
    struct TMakeDataProviderFromTextOptions {
        // Space is chosen as a default separator instead of tab because it is far more readable
        // (when data is small) and it is a lot more easier to write test cases when columns are
        // space separated;
        char Delimiter = ' ';

        // Lines are separated by '\n' by default, but in case you want change it here is a
        // parameter for you.
        char LineSeparator = '\n';

        // When writing raw string within a .cpp we may sometimes acidentally put spaces in the
        // beginning and in the end of the line; To avoid any stupid mistakes we strip spaces by
        // default.
        bool StripSpacesLeft = true;
        bool StripSpacesRight = true;
    };

    // Create dataset from text.
    //
    // @param columnsDescription        Columns description as a string, has same format as .cd
    //                                  files.
    // @param dataset                   Dataset in same format as accepted by catboost cli tool.
    //
    // @returns                         Dataset provider.
    //
    // Example:
    // ```
    //     MakeDataProviderFromTest(
    //         "0\tLabel\n",
    //         "1\tNum\n",
    //         R"(
    //             xxx 11
    //             yyy 12
    //         )");
    // ```
    //
    // Will create a dataset containing two objects with labels "xxx" first one and "yyy" for the
    // second one. Each object will have one numeric feature (with value 11 for the forst object and
    // value 12 for the second object).
    //
    // Note that for columns description we had to use tabs, that is because column description file
    // is not customizable. But DSV parser is, by default we also ignore whitespaces in the
    // beginning and in the end of the dataset and use space as a separator, not a tabulation.
    //
    // TODO(yazevnul): support pairs
    TDataProviderPtr MakeDataProviderFromText(
        TStringBuf columnsDescription,
        TStringBuf dataset,
        const TMakeDataProviderFromTextOptions& options = {});
}
