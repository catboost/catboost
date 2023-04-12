#include <library/cpp/archive/yarchive.h>

#include <util/generic/buffer.h>
#include <util/generic/yexception.h>
#include <util/generic/singleton.h>

#include <util/system/hostname.h>
#include <util/memory/blob.h>
#include <util/stream/output.h>
#include <util/stream/buffer.h>

//lua code, with hooks
struct lua_State;

static void RegisterYandexHooks(lua_State* l);
static int loader_Common(lua_State* L);

#define LUA_LIB
#define luaall_c

#define ltable_c    // for luai_hashnum under Visual Studio

#include "lobject.c"
#include "lstate.c"
#include "lgc.c"
#include "lvm.c"
#include "lauxlib.c"
#include "ltm.c"
#include "lcode.c"
#include "lapi.c"
#include "lmathlib.c"
#include "lparser.c"
#include "ldblib.c"
#include "ldebug.c"
#include "lzio.c"
#include "loslib.c"
#include "ltable.c"
#include "lmem.c"
#include "liolib.c"
#include "lstring.c"
#include "lundump.c"
#include "linit.c"
#include "llex.c"
#include "lopcodes.c"
#include "lstrlib.c"
#include "lfunc.c"
#include "ldo.c"
#include "lbaselib.c"
#include "ldump.c"
#include "ltablib.c"
#include "loadlib.c"
#include "lutf8lib.c"

#if LUA_VERSION_NUM >= 502
    #include "lbitlib.c"
    #include "lctype.c"
    #include "lcorolib.c"
#endif

//end of lua code

//yandex stuff
namespace {
    static const unsigned char data[] = {
         #include "common.inc"
    };

    class TLuaArchive: public TArchiveReader {
    public:
        inline TLuaArchive()
            : TArchiveReader(TBlob::NoCopy(data, sizeof(data)))
        {
        }
    };

    static inline TArchiveReader& Archive() {
        return *Singleton<TLuaArchive>();
    }

    static inline bool OpenInternalModule(const char* key, TBuffer& buf) {
        try {
            TAutoPtr<IInputStream> in(Archive().ObjectByKey(key));
            TBufferOutput out(buf);

            TransferData(in.Get(), &out);

            return true;
        } catch (...) {
        }

        return false;
    }
}

#if LUA_VERSION_NUM == 502
static int loader_Common(lua_State* L) {
    const char* extname;
    char filename[256];

    extname = luaL_checkstring(L, 1);
    snprintf(filename, 256, "/%s.lua", extname);

    TBuffer buf;
    if (!OpenInternalModule(filename, buf)) {
        return 1;
    }

    if (luaL_loadbuffer(L, buf.Data(), buf.Size(), filename) != 0) {
        return 1;
    }

    lua_pushstring(L, filename);

    return 2;
}
#endif

#if LUA_VERSION_NUM == 501
static int loader_Common(lua_State* L) {
    const char* extname;
    char filename[256];

    extname = luaL_checkstring(L, 1);
    snprintf(filename, 256, "/%s.lua", extname);

    TBuffer buf;
    if (!OpenInternalModule(filename, buf)) {
        loaderror(L, extname);

        return 1;
    }

    if (luaL_loadbuffer(L, buf.Data(), buf.Size(), filename) != 0) {
        loaderror(L, filename);
    }

    return 1;  /* library loaded successfully */
}
#endif

static int OsHostName(lua_State* l) {
    lua_pushstring(l, HostName().data());

    return 1;
}

static int OsPutEnv(lua_State* l) {
    const char* env = luaL_checkstring(l, 1);
    int res = putenv(strdup(env));
    lua_pushnumber(l, res);

    return 1;
}

static const luaL_Reg YANDEX_OS_LIBS[] = {
    {"hostname", OsHostName},
    {"putenv", OsPutEnv},
    {0, 0}
};

LUALIB_API int RegisterYandexOsHooks(lua_State* l) {
    luaL_register(l, LUA_OSLIBNAME, YANDEX_OS_LIBS);

    return 1;
}

static void RegisterYandexHooks(lua_State* l) {
    lua_pushcfunction(l, RegisterYandexOsHooks);
    lua_pushstring(l, "RegisterYandexOsHooks");
    lua_call(l, 1, 0);
}
