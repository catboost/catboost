#include <library/cpp/testing/unittest/utmain.h>
#include <catboost/cuda/cuda_lib/mpi/mpi_manager.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>

int main(int argc, char** argv) {
    bool isSlave = false;

#if defined(USE_MPI)
    CB_ENSURE(argc);
    auto& mpiManager = NCudaLib::GetMpiManager();
    mpiManager.Start(&argc, &argv);
    if (!mpiManager.IsMaster()) {
        isSlave = true;
        RunSlave();
    }
#endif
    auto& config = NCudaLib::GetDefaultDeviceRequestConfig();
    config.PinnedMemorySize = ((ui64)4) * 1024 * 1024 * 1024;

    int exitCode = 0;
    if (!isSlave) {
        exitCode = NUnitTest::RunMain(argc, argv);
    }

#if defined(USE_MPI)
    //ensure cudaManager was started at least once
    if (!isSlave) {
        auto stopGuard = StartCudaManager();
    }
    if (mpiManager.IsMaster()) {
        mpiManager.Stop();
    }
#endif
    return exitCode;
}
