#include <catboost/private/libs/data_types/groupid.h>

#include <util/generic/algorithm.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/stream/fwd.h>
#include <util/stream/file.h>
#include <util/stream/output.h>
#include <util/string/cast.h>
#include <util/string/split.h>
#include <util/system/types.h>

#include <utility>


int main(int argc, const char* argv[]) {
    if (argc != 4) {
        Cerr << "wrong number of input arguments\n"
        "arg1 is 0-based field index\n"
        "arg2 is an input file\n"
        "arg3 is an output file\n";

        return -1;
    }

    auto fieldIdx = FromString<ui32>(argv[1]);

    // numeric group id is i64 to sort as in Spark
    TVector<std::pair<TString, i64>> lines;
    {
        TIFStream in(argv[2]);
        TString line;
        while (in.ReadLine(line)) {
            TVector<TStringBuf> fields = StringSplitter(line).Split('\t');
            auto groupId = (i64)CalcGroupIdFor(fields[fieldIdx]);
            lines.push_back(std::make_pair(std::move(line), groupId));
        }
    }

    StableSort(lines, [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });

    {
        TOFStream out(argv[3]);
        for (const auto& [line, groupId] : lines) {
            out << line << '\n';
        }
    }

    return 0;
}
