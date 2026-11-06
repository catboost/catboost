// Fallback definition of the static resource table referenced from registry.cpp: an empty table
// (no magic, so it is ignored). library/python/runtime_py3/pyres_ld_plugin.py puts an object with
// the real table in front of all the libraries on the link command line; then the linker never
// pulls this object out of the archive. It must stay alone in its translation unit for that.
extern "C" alignas(16) const char __pyres_static_blob[16] = {};
