Hierarchical Navigable Small World graphs implementation
=====================================================
Please refer to original paper for details on the algorithm:
https://arxiv.org/ftp/arxiv/papers/1603/1603.09320.pdf

This library comprises following directories:

* `index` - library for querying HNSW index
    * `index_base.h` contains the base class for your own custom indexes
    * `dense_vector_index.h` contains implementation for most common case of searching in a set of N-dimensional dense vectors
* `index_builder` - library for building HNSW index
    * `index_builder.h` contains method BuildIndex for building your own custom indexes
    * `dense_vector_index_builder.h` contains method BuildDenseVectorIndex for building commonly used indexes over set N-dimensional dense vectors
* `tools` - assorted tools for working with HNSW
    * `build_dense_vector_index` - basic self-explanatory program for building dense vector indexes
    * `measure_recall` - tool for evaluating requests-per-second and Recall@k quality metric on a custom HNSW index and a query bucket

Please refer to the `ut/main.cpp` for a comprehensive tutorial on how to build and search your own custom index.

Some discussion can be found in SAAS-2991.
