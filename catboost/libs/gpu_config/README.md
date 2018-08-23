If you want to use CUDA if it is available and not disabled by command-line build flag (HAVE_CUDA) set PEERDIR to catboost/libs/gpu_config/maybe_have_cuda.
If you want to disable CUDA support even if it is available independently of HAVE_CUDA flag (if you want to build CUDA-disabled along with CUDA-enabled version in one build)
PEERDIR to catboost/gpu_config/force_no_cuda.

In both cases add 
```cpp
#include <catboost/libs/gpu_config/interface/get_gpu_device_count.h>
```


