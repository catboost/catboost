#if defined(__clang__) && __clang_major__ * 100 + __clang_minor__ < 309
extern "C" char* _ZTVSt9type_info;

struct TFakeTI {
    void* VTBL;
    const char* Name;
};

extern "C" const TFakeTI _ZTIn = {(char*)&_ZTVSt9type_info + 16, "n"};
extern "C" const TFakeTI _ZTIo = {(char*)&_ZTVSt9type_info + 16, "o"};
#endif
