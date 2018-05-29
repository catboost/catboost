#include "line_data_reader.h"

#include <catboost/libs/helpers/exception.h>

#include <util/stream/file.h>
#include <util/system/fs.h>


namespace NCB {

    THolder<ILineDataReader> GetLineDataReader(const TPathWithScheme& pathWithScheme,
                                               const TDsvFormatOptions& format)
    {
        return GetProcessor<ILineDataReader, TLineDataReaderArgs>(
            pathWithScheme, TLineDataReaderArgs{pathWithScheme, format}
        );
    }


    namespace {

    template <class TStr>
    inline int CountLines(const TStr& poolFile) {
        CB_ENSURE(NFs::Exists(TString(poolFile)), "pool file '" << TString(poolFile) << "' is not found");
        TIFStream reader(poolFile.c_str());
        size_t count = 0;
        TString buffer;
        while (reader.ReadLine(buffer)) {
            ++count;
        }
        return count;
    }

    class TFileLineDataReader : public ILineDataReader {
    public:
        TFileLineDataReader(const TLineDataReaderArgs& args)
            : Args(args)
            , IFStream(args.PathWithScheme.Path)
            , HeaderProcessed(!Args.Format.HasHeader)
        {}

        ui64 GetDataLineCount() override {
            ui64 nLines = (ui64)CountLines(Args.PathWithScheme.Path);
            if (Args.Format.HasHeader) {
                --nLines;
            }
            return nLines;
        }

        TMaybe<TString> GetHeader() override {
            if (Args.Format.HasHeader) {
                CB_ENSURE(!HeaderProcessed, "TFileLineDataReader: multiple calls to GetHeader");
                TString header;
                CB_ENSURE(IFStream.ReadLine(header), "TFileLineDataReader: no header in file");
                HeaderProcessed = true;
                return header;
            }

            return {};
        }

        bool ReadLine(TString* line) override {
            // skip header if it hasn't been read
            if (!HeaderProcessed) {
                GetHeader();
            }
            return IFStream.ReadLine(*line) != 0;
        }

    private:
        TLineDataReaderArgs Args;
        TIFStream IFStream;
        bool HeaderProcessed;
    };


    TLineDataReaderFactory::TRegistrator<TFileLineDataReader> DefLineDataReaderReg("");
    TLineDataReaderFactory::TRegistrator<TFileLineDataReader> FileLineDataReaderReg("file");
    TLineDataReaderFactory::TRegistrator<TFileLineDataReader> DsvLineDataReaderReg("dsv");

    }
}
