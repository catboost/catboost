#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

constexpr char DELIMITER = '\t';

static vector<string> Tokenize(const string& line, char delimiter) {
    vector<string> result;
    string token;
    istringstream input(line);
    while (getline(input, token, delimiter)) {
        result.push_back(token);
    }
    return result;
}

static void ReadColumnDescription(const string& cdFile, vector<size_t>* floatColumns, vector<size_t>* catColumns, set<size_t>* otherColumns) {
    ifstream cd(cdFile);
    string line;
    while (getline(cd, line)) {
        const vector<string>& tokens = Tokenize(line, DELIMITER);
        assert(tokens.size() >= 2);

        size_t colNo = stoul(tokens[0]);
        const string& colDescription = tokens[1];

        if (colDescription == "Categ") {
            catColumns->push_back(colNo);
        } else if (colDescription == "Num") {
            floatColumns->push_back(colNo);
        } else {
            otherColumns->insert(colNo);
        }
    }
}

static void AdjustFloatColumns(size_t columnCount, const vector<size_t>& catColumns, const set<size_t>& otherColumns, vector<size_t>* floatColumns) {
    size_t iFloat = 0;
    size_t iCat = 0;
    for (size_t i = 0; i < columnCount; ++i) {
        // Figure out the type of the ith column
        if (otherColumns.find(i) != otherColumns.end()) {
            continue;
        }
        if (iCat < catColumns.size() && catColumns[iCat] == i) {
            iCat++;
            continue;
        }
        if (iFloat < floatColumns->size() && (*floatColumns)[iFloat] == i) {
            iFloat++;
            continue;
        }
        // Now the ith column is implicit numerical feature,
        // therefore make it listed explicitly in floatColumns
        if (iFloat < floatColumns->size()) {
            assert(i < (*floatColumns)[iFloat]);
            floatColumns->insert(floatColumns->begin() + iFloat, i);
            iFloat++;
            continue;
        } else {
            floatColumns->push_back(i);
            iFloat++;
            continue;
        }
    }
}

static void ParseFeatures(
    const string& line,
    const vector<size_t>& floatColumns,
    const vector<size_t>& catColumns,
    vector<float>* floatFeatures,
    vector<string>* catFeatures) {

    const vector<string>& tokens = Tokenize(line, DELIMITER);
    for (size_t i = 0; i < floatColumns.size(); ++i) {
        floatFeatures->push_back(stof(tokens[floatColumns[i]]));
    }
    for (size_t i = 0; i < catColumns.size(); ++i) {
        catFeatures->push_back(tokens[catColumns[i]]);
    }
}

extern std::vector<double> ApplyCatboostModelMulti(const vector<float>& floatFeatures, const vector<string>& catFeatures);

int main(int argc, char *argv[]) {
    assert((argc == 4) || (argc == 5));  // main.exe test.tsv cd.tsv predictions.txt [class_names]

    vector<string> classNames;
    if (argc == 5) {
        classNames = Tokenize(argv[4], ',');
    }

    vector<size_t> floatColumns, catColumns;
    set<size_t> otherColumns;
    ReadColumnDescription(argv[2], &floatColumns, &catColumns, &otherColumns);
    sort(floatColumns.begin(), floatColumns.end());
    sort(catColumns.begin(), catColumns.end());

    ifstream test(argv[1]);
    ofstream predictions(argv[3]);
    predictions << setprecision(10);
    string line;
    for (size_t docId = 0; getline(test, line); ++docId) {
        vector<float> floatFeatures;
        vector<string> catFeatures;
        if (docId == 0) {
            // Column description may not mention all columns,
            // so the actual number of columns is not known up to this point.
            size_t columnCount = 1 + count(line.begin(), line.end(), DELIMITER);
            AdjustFloatColumns(columnCount, catColumns, otherColumns, &floatColumns);
        }
        ParseFeatures(line, floatColumns, catColumns, &floatFeatures, &catFeatures);

        auto rawFormulaVals = ApplyCatboostModelMulti(floatFeatures, catFeatures);

        if (docId == 0) {
            predictions << "SampleId";
            if (rawFormulaVals.size() == 1) {
                predictions << DELIMITER << "RawFormulaVal";
            } else {
                assert(rawFormulaVals.size() == classNames.size());

                // Multiclassification
                for (size_t classIdx = 0; classIdx < rawFormulaVals.size(); ++classIdx) {
                    predictions << DELIMITER << "RawFormulaVal:Class=" << classNames[classIdx];
                }
            }
            predictions << endl;
        }

        predictions << docId;
        for (size_t predIdx = 0; predIdx < rawFormulaVals.size(); ++predIdx) {
            predictions << DELIMITER << rawFormulaVals[predIdx];
        }
        predictions << endl;
    }

    return 0;
}
