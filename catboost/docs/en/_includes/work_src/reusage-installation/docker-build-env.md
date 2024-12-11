{% note info %}

There are Linux Ubuntu 22.04 - based docker images with the build environment set up.
They work only on x86_64 processor architecture for now.

Two variations:
  - `ghcr.io/catboost/build_env_no_cuda:ubuntu_22.04` - without CUDA (`base` target in docker file)
  - `ghcr.io/catboost/build_env_with_cuda:ubuntu_22.04` - with CUDA (`with_cuda` target in docker file)

They are built from a dockerfile [here](https://github.com/catboost/catboost/tree/master/build/docker) so you can also modify it for your needs.

{% endnote %}
