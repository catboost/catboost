## Docker-based build environments

Ubuntu 22.04 - based.
Work only on x86_64 processor architecture for now.

Two variations:
  - `ghcr.io/catboost/build_env_no_cuda:ubuntu_22.04` - without CUDA (`base` target in docker file)
  - `ghcr.io/catboost/build_env_with_cuda:ubuntu_22.04` - with CUDA (`with_cuda` target in docker file)

Run `build.sh` to build docker images with standard names.
