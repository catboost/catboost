# TODO: adapt for non-x86_64 architectures

FROM ubuntu:22.04 AS base

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    gpg \
    wget \
    curl \
    git \
    build-essential \
    software-properties-common \
    libstdc++-12-dev \
    python3-pip

# Install recent GCC (GCC 12 in Ubuntu 22.04)
RUN apt-get install -y gcc-12 g++-12 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# Install GDB
RUN apt-get install -y gdb

# Install Clang 16, make it the default version
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 16 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-16 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-16 100 && \
    rm llvm.sh

# Some code uses 'python' command so we need it to be available
RUN apt-get install -y python-is-python3

# Update pip
RUN pip3 install --upgrade pip

# Update setuptools, wheel and build
RUN pip3 install --upgrade 'setuptools>=64.0' wheel build

# Install jupyterlab for catboost-widget
RUN pip3 install jupyterlab==3.0.6

# Install CMake >= 3.24
RUN pip3 install 'cmake>=3.24.0'

# Install Conan 2.4.1
RUN pip3 install conan==2.4.1

RUN pip3 install cython==3.0.12

RUN pip3 install numpy

# Install Ninja
RUN pip3 install ninja

# Install Node, npm, rimraf and yarn (needed for CatBoost python package visualization widget and Node package)
ENV NODE_VERSION 18

SHELL ["/bin/bash", "--login", "-i", "-c"]
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
RUN source /root/.bashrc && \
    nvm install $NODE_VERSION && \
    nvm alias default $NODE_VERSION && \
    nvm use default && \
    npm install --global yarn@1.22.10 rimraf

# Install R and 'devtools' package
RUN apt-get update && \
    # these packages are needed for 'devtools' package with dependencies
    apt-get -y install libcurl4-gnutls-dev libxml2-dev libssl-dev libfontconfig1-dev libharfbuzz-dev libfribidi-dev libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev && \
    apt-get -y install r-base r-base-dev && \
    Rscript -e "install.packages('devtools')"

# Install JDK8, make it the default version
RUN wget -qO - https://packages.adoptium.net/artifactory/api/gpg/key/public | gpg --dearmor | tee /etc/apt/trusted.gpg.d/adoptium.gpg > /dev/null
RUN echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" | tee /etc/apt/sources.list.d/adoptium.list
RUN apt-get update && \
    apt-get install -y temurin-8-jdk

ENV JAVA_HOME=/usr/lib/jvm/temurin-8-jdk-amd64/

RUN update-alternatives --install /usr/bin/java java ${JAVA_HOME}/bin/java 100 && \
    update-alternatives --install /usr/bin/javac javac ${JAVA_HOME}/bin/javac 100

# Install Maven
RUN apt-get update && \
    apt-get -y install maven

# Install Cargo for Rust package
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

# Verify installations
RUN gcc --version && \
    gdb --version && \
    clang --version && \
    jupyter --version && \
    cmake --version && \
    conan --version && \
    ninja --version && \
    node --version && \
    npm --version && \
    yarn --version && \
    R --version && \
    java -version && \
    javac -version && \
    mvn -version && \
    cargo --version

# Cleanup
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists

# Set the default shell to Bash
CMD ["/bin/bash"]


FROM base AS with_cuda

# Install Clang 14 (to be used as a CUDA host compiler)
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 14 && \
    rm llvm.sh

# Install CUDA Toolkit 11.8
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get -y install cuda-11-8

# Set environment variables for CUDA
ENV CUDA_ROOT=/usr/local/cuda-11.8/
ENV PATH="${CUDA_ROOT}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH}"

# Verify installation
RUN clang-14 --version && \
    nvcc --version

# Cleanup
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists

# Set the default shell to Bash
CMD ["/bin/bash"]
