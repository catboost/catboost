// Fixed standard library, system allocator and abort-on-panic runtime.
// Application crates are linked separately through their native archives.

// Target libraries such as syn/proc-macro2 also reference standard-library
// monomorphizations exported by proc_macro, even outside the compiler process.
extern crate proc_macro;
