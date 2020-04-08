#pragma once

#include "exception.h"

#include <catboost/libs/logging/logging.h>

#include <util/stream/output.h>
#include <util/stream/file.h>
#include <util/folder/path.h>
#include <util/generic/guid.h>
#include <util/system/fs.h>
#include <util/ysaveload.h>

#include <library/cpp/digest/md5/md5.h>

class TMD5Output : public IOutputStream {
public:
    explicit inline TMD5Output(IOutputStream* slave) noexcept
        : Slave_(slave) {
    }

    inline const char* Sum(char* buf) {
        return MD5Sum_.End(buf);
    }

private:
    void DoWrite(const void* buf, size_t len) override {
        Slave_->Write(buf, len);
        MD5Sum_.Update(buf, len);
    }

    /* Note that default implementation of DoSkip works perfectly fine here as
        * it's implemented in terms of DoRead. */

private:
    IOutputStream* Slave_;
    MD5 MD5Sum_;
};

class TProgressHelper {
public:
    explicit TProgressHelper(
        const TString& label,
        const TString exceptionMessage = "Can't save progress to file, got exception: ",
        const TString savedMessage = "Saved progress",
        bool calcMd5 = true
    )
        : Label(label)
        , ExceptionMessage(exceptionMessage)
        , SavedMessage(savedMessage)
        , CalcMd5(calcMd5)
    {}

    template <class TWriter>
    void Write(const TFsPath& path, TWriter&& writer) {
        TString tempName = JoinFsPaths(path.Dirname(), CreateGuidAsString()) + ".tmp";
        try {
            {
                TOFStream out(tempName);
                TMD5Output md5out(&out);
                ::Save(&md5out, Label);
                writer(&md5out);
                char md5buf[33];
                if (CalcMd5) {
                    CATBOOST_INFO_LOG << SavedMessage << " (md5sum: " << md5out.Sum(md5buf) << " )" << Endl;
                }
            }
            NFs::Rename(tempName, path);
        } catch (...) {
            CATBOOST_WARNING_LOG << ExceptionMessage <<  CurrentExceptionMessage() << Endl;
            NFs::Remove(tempName);
        }
    }

    template <class TReader>
    void CheckedLoad(const TFsPath& path, TReader&& reader) {
        TString label;
        TIFStream input(path);
        ::Load(&input, label);
        CB_ENSURE(Label == label, "Error: expect " << Label << " progress. Got " << label);
        reader(&input);
    }

private:
    TString Label;
    TString ExceptionMessage;
    TString SavedMessage;
    bool CalcMd5;
};
