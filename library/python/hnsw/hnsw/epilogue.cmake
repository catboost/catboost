
target_include_directories(_hnsw PRIVATE
  ${Python3_NumPy_INCLUDE_DIRS}
)

target_cython_include_directories(_hnsw
  ${Python3_NumPy_INCLUDE_DIRS}
)
