add_link_options(
  -nodefaultlibs
  -lc
  -lm
)

if (APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "-undefined dynamic_lookup")
endif()
