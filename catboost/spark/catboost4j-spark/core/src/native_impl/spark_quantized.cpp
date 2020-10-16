#include <catboost/private/libs/data_util/exists_checker.h>
#include <catboost/private/libs/quantized_pool/loader.h>

#include <catboost/libs/data/loader.h>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/ptr.h>
#include <util/system/fs.h>


using namespace NCB;


/**
 * Path in scheme "spark-quantized" has the following structure:
 *  "master-part:<path_to_quantized_pool>"
 *
 *  <path_to_quantized_pool> - path to local file in "quantized" format.
 *    It typically has features data for test parts and no features data for train part
 */

static TPathWithScheme GetMasterPartPath(const TPathWithScheme& poolPath) {
    constexpr TStringBuf MASTER_PART = "master-part:";

    CB_ENSURE(
        poolPath.Path.StartsWith(MASTER_PART),
        "Pool path does not contain \"" << MASTER_PART << "\" prefix"
    );

    TPathWithScheme masterPart;
    masterPart.Scheme = "quantized";
    masterPart.Path = poolPath.Path.substr(MASTER_PART.size(), poolPath.Path.size() - MASTER_PART.size());

    return masterPart;
}

struct TSparkQuantizedFSExistsChecker : public IExistsChecker {
    bool Exists(const TPathWithScheme& pathWithScheme) const override {
        return NFs::Exists(GetMasterPartPath(pathWithScheme).Path);
    }

    bool IsSharedFs() const override {
        return true;
    }
};


class TSparkQuantizedMasterDataLoader : public IQuantizedFeaturesDatasetLoader {
public:
    explicit TSparkQuantizedMasterDataLoader(TDatasetLoaderPullArgs&& args) {
        args.PoolPath = GetMasterPartPath(args.PoolPath);
        MasterDataLoader = MakeHolder<TCBQuantizedDataLoader>(std::move(args));
    }

    void Do(IQuantizedFeaturesDataVisitor* visitor) override {
        MasterDataLoader->Do(visitor);
    }

private:
    THolder<TCBQuantizedDataLoader> MasterDataLoader;
};


namespace {
    TExistsCheckerFactory::TRegistrator<TSparkQuantizedFSExistsChecker> SparkQuantizedExistsCheckerReg("spark-quantized");
    TDatasetLoaderFactory::TRegistrator<TSparkQuantizedMasterDataLoader> SparkQuantizedDataLoaderReg("spark-quantized");
}
