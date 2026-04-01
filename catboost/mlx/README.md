# CatBoost-MLX: GPU Training on Apple Silicon

CatBoost-MLX replaces the CUDA GPU backend with Apple's Metal via the [MLX](https://github.com/ml-explore/mlx) framework, enabling gradient boosted decision tree training on Apple Silicon Macs (M1/M2/M3/M4).

## Prerequisites

| Requirement | Minimum |
|------------|---------|
| **macOS** | 14.0 (Sonoma) |
| **Hardware** | Apple Silicon (M1, M2, M3, M4) |
| **Xcode** | 15.0+ (for Metal compiler) |
| **MLX** | 0.22+ (C++ library) |

## Installing MLX

MLX is Apple's open-source array framework for Metal GPU computation. It is the only external dependency.

### Option 1: Homebrew (recommended)

```bash
brew install mlx
```

Verify installation:
```bash
brew info mlx
# Should show: mlx: stable X.Y.Z, installed at /opt/homebrew/Cellar/mlx/X.Y.Z
```

### Option 2: Build from source

```bash
git clone https://github.com/ml-explore/mlx.git
cd mlx
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF
make -j$(sysctl -n hw.ncpu)
sudo make install
```

### Finding MLX paths

After installation, locate include and library paths:

```bash
# Homebrew (Apple Silicon)
MLX_INCLUDE=$(brew --prefix mlx)/include
MLX_LIB=$(brew --prefix mlx)/lib

# Verify
ls $MLX_INCLUDE/mlx/mlx.h
ls $MLX_LIB/libmlx.*
```

## Building the Standalone CSV Training Tool

The standalone `csv_train` binary can train GBDT models from any CSV file using the Metal GPU.

```bash
cd catboost-mlx

# Auto-detect MLX paths
MLX_PREFIX=$(brew --prefix mlx)

clang++ -std=c++17 -O2 -I. \
  -I${MLX_PREFIX}/include \
  -L${MLX_PREFIX}/lib -lmlx \
  -framework Metal -framework Foundation \
  -Wno-c++20-extensions \
  catboost/mlx/tests/csv_train.cpp -o csv_train
```

### Quick test

```bash
# Create a small test file
cat > test.csv << 'EOF'
x1,x2,target
0.1,0.2,0
0.3,0.1,0
0.6,0.7,1
0.8,0.9,1
0.9,0.8,1
0.2,0.3,0
EOF

./csv_train test.csv --loss logloss --iterations 50 --verbose
```

## Usage

```
./csv_train <file.csv> [options]

Options:
  --iterations N      Number of boosting iterations (default: 100)
  --depth D           Max tree depth (default: 6)
  --lr RATE           Learning rate (default: 0.1)
  --l2 LAMBDA         L2 regularization (default: 3.0)
  --loss TYPE         Loss: rmse, logloss, multiclass, auto (default: auto)
  --bins B            Max quantization bins per feature (default: 255)
  --target-col N      0-based column index for target (default: last column)
  --cat-features L    Comma-separated 0-based column indices for categorical features
  --verbose           Print per-iteration loss
```

### Loss auto-detection

If `--loss auto` (default), the tool detects the task type from the target column:
- **{0, 1}** targets → `logloss` (binary classification)
- **{0, 1, ..., K-1}** with K > 2 → `multiclass`
- **continuous** → `rmse` (regression)

### Categorical features

Columns with non-numeric values are auto-detected as categorical. You can also specify them explicitly:

```bash
# Auto-detect (non-numeric columns treated as categorical)
./csv_train titanic.csv --loss logloss --iterations 200

# Explicit: columns 0 and 2 are categorical
./csv_train data.csv --cat-features 0,2 --loss logloss
```

Categorical features use OneHot encoding: each unique category value becomes a split candidate via equality comparison (`value == category`), matching CatBoost's CUDA behavior.

### Examples

```bash
# Regression
./csv_train housing.csv --loss rmse --depth 4 --lr 0.05 --iterations 500

# Binary classification
./csv_train fraud.csv --loss logloss --iterations 200 --verbose

# Multiclass (e.g., Iris dataset)
./csv_train iris.csv --loss multiclass --iterations 300

# Mixed categorical + numeric
./csv_train customer_churn.csv --cat-features 1,3,5 --loss logloss --iterations 200
```

## Building the Full CatBoost-MLX Library

To build CatBoost with the MLX GPU backend (instead of CUDA):

```bash
cd catboost-mlx
mkdir build && cd build
cmake .. -DUSE_MLX=ON -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

This requires the full CatBoost build system (ya make or CMake) and is more involved than the standalone tool.

## Architecture

```
catboost/mlx/
├── kernels/           # Metal Shading Language GPU kernels
│   ├── kernel_sources.h  # Histogram kernel (one-byte features)
│   ├── hist.metal        # Additional histogram kernels
│   ├── leaves.metal      # Leaf value computation
│   └── scores.metal      # Split scoring
├── gpu_data/          # GPU data structures
│   ├── gpu_structures.h  # TCFeature, split descriptors
│   ├── mlx_device.h      # MLX device wrapper
│   └── data_set_builder.cpp
├── methods/           # Training algorithms
│   ├── score_calcer.cpp  # Split scoring (OneHot + ordinal)
│   ├── structure_searcher.cpp  # Greedy tree search
│   ├── tree_applier.cpp  # Tree evaluation
│   └── histogram.cpp     # GPU histogram dispatch
├── targets/           # Loss functions (RMSE, Logloss, MultiClass)
├── train_lib/         # Training loop orchestration
└── tests/             # Test files
    └── csv_train.cpp     # Standalone CSV training tool
```

## Troubleshooting

### "Cannot find mlx/mlx.h"

MLX is not installed or not on the include path. Install via `brew install mlx` and verify:
```bash
ls $(brew --prefix mlx)/include/mlx/mlx.h
```

### "Library not found for -lmlx"

MLX library not on the library path. Check:
```bash
ls $(brew --prefix mlx)/lib/libmlx*
```

If using a custom MLX build, set paths manually:
```bash
clang++ ... -I/path/to/mlx/include -L/path/to/mlx/lib -lmlx ...
```

### "Undefined symbols for architecture arm64"

Ensure you're compiling on Apple Silicon (not Intel via Rosetta):
```bash
uname -m  # should print: arm64
```

### Runtime: "No Metal device found"

Requires a Mac with Apple Silicon. Verify:
```bash
system_profiler SPDisplaysDataType | grep "Metal Support"
```

### Slow first iteration

Metal shader compilation happens on the first kernel dispatch. Subsequent iterations are faster.
