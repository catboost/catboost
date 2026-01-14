# CatBoost Project Overview
CatBoost is a high-performance gradient boosting library developed by Yandex.
It specializes in handling both numerical and categorical features natively.
## Key Features
- Superior prediction quality on many datasets
- Best-in-class prediction speed with C++ API
- Native support for categorical features without preprocessing
- GPU and multi-GPU training support out of the box
- Reproducible distributed training with Apache Spark
- Built-in visualization tools for model analysis
## Project Structure
The codebase is organized into several main directories:
- `catboost/app/` - CLI applications for training, evaluation, and feature selection
- `catboost/libs/` - Core C++ libraries implementing the gradient boosting engine
- `catboost/python-package/` - Python bindings and scikit-learn compatible API
- `catboost/cuda/` - GPU/CUDA acceleration support
- `catboost/jvm-packages/` - Java and Scala bindings
- `catboost/R-package/` - R language bindings
- `catboost/rust-package/` - Rust bindings
- `catboost/spark/` - Apache Spark integration for distributed training
## Core Components
### Python Package
- `core.py` - Main classes: CatBoostClassifier, CatBoostRegressor, CatBoostRanker
- `_catboost.pyx` - Cython bindings to the C++ core engine
- `utils.py` - Utility functions for data processing
### C++ Libraries
- `calc_metrics/` - Metric calculation engine
- `cat_feature/` - Categorical feature handling
- `fstr/` - Feature importance computation
- `model/` - Model core classes and serialization
- `train_lib/` - Training algorithms implementation
## Supported Platforms
- Linux (x86_64, aarch64, ppc64le)
- macOS (x86_64, arm64)
- Windows
- Android
## Build System
The project uses CMake with platform-specific configurations.
CUDA support is available for GPU acceleration.
Cython is used for Python C++ bindings.
## License
Apache 2.0 License by Yandex LLC.
