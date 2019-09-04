#include "line_data_reader.h"

#include <util/system/fs.h>


namespace NCB {

    THolder<ILineDataReader> GetLineDataReader(const TPathWithScheme& pathWithScheme,
                                               const TDsvFormatOptions& format)
    {
        return GetProcessor<ILineDataReader, TLineDataReaderArgs>(
            pathWithScheme, TLineDataReaderArgs{pathWithScheme, format}
        );
    }

    int CountLines(const TString& poolFile) {
        CB_ENSURE(NFs::Exists(TString(poolFile)), "pool file '" << TString(poolFile) << "' is not found");
        TIFStream reader(poolFile.c_str());
        size_t count = 0;
        TString buffer;
        while (reader.ReadLine(buffer)) {
            ++count;
        }
        return count;
    }

    TLineDataReaderFactory::TRegistrator<TFileLineDataReader> DefLineDataReaderReg("");
    TLineDataReaderFactory::TRegistrator<TFileLineDataReader> FileLineDataReaderReg("file");
    TLineDataReaderFactory::TRegistrator<TFileLineDataReader> DsvLineDataReaderReg("dsv");
}
