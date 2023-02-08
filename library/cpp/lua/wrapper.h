#pragma once

#include <library/cpp/string_utils/ztstrbuf/ztstrbuf.h>

#include <util/memory/alloc.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/yexception.h>
#include <util/generic/buffer.h>
#include <util/datetime/base.h>
#include <functional>

#include <contrib/libs/lua/lua.h>

class IInputStream;
class IOutputStream;

class TLuaStateHolder {
    struct TDeleteState {
        static inline void Destroy(lua_State* state) {
            lua_close(state);
        }
    };

public:
    class TError: public yexception {
    };

    inline TLuaStateHolder(size_t memory_limit = 0)
        : AllocFree(memory_limit)
        , MyState_(lua_newstate(memory_limit ? AllocLimit : Alloc, (void*)this))
        , State_(MyState_.Get())
    {
        if (!State_) {
            ythrow TError() << "can not construct lua state: not enough memory";
        }
    }

    inline TLuaStateHolder(lua_State* state) noexcept
        : State_(state)
    {
    }

    inline operator lua_State*() noexcept {
        return State_;
    }

    inline void BootStrap() {
        luaL_openlibs(State_);
    }

    inline void error() {
        ythrow TError() << "lua error: " << pop_string();
    }

    inline bool is_string(int index) {
        return lua_isstring(State_, index);
    }

    inline void is_string_strict(int index) {
        if (!is_string(index)) {
            ythrow TError() << "internal lua error (not a string)";
        }
    }

    inline TStringBuf to_string(int index) {
        size_t len = 0;
        const char* data = lua_tolstring(State_, index, &len);
        return TStringBuf(data, len);
    }

    inline TStringBuf to_string(int index, TStringBuf defaultValue) {
        return is_string(index) ? to_string(index) : defaultValue;
    }

    inline TStringBuf to_string_strict(int index) {
        is_string_strict(index);
        return to_string(index);
    }

    inline TString pop_string() {
        TString ret(to_string(-1));
        pop();
        return ret;
    }

    inline TString pop_string(TStringBuf defaultValue) {
        TString ret(to_string(-1, defaultValue));
        pop();
        return ret;
    }

    inline TString pop_string_strict() {
        require(1);
        TString ret(to_string_strict(-1));
        pop();
        return ret;
    }

    inline TString pop_value() {
        require(1);
        if (is_bool(-1)) {
            return pop_bool() ? "true" : "false";
        }
        return pop_string_strict();
    }

    inline void push_string(const char* st) {
        lua_pushstring(State_, st ? st : "");
    }

    inline void push_string(TStringBuf st) {
        lua_pushlstring(State_, st.data(), st.size());
    }

    inline bool is_number(int index) {
        return lua_isnumber(State_, index);
    }

    inline void is_number_strict(int index) {
        if (!is_number(index)) {
            ythrow TError() << "internal lua error (not a number)";
        }
    }

    template <typename T>
    inline T to_number(int index) {
        return static_cast<T>(lua_tonumber(State_, index));
    }

    template <typename T>
    inline T to_number(int index, T defaultValue) {
        return is_number(index) ? to_number<T>(index) : defaultValue;
    }

    template <typename T>
    inline T to_number_strict(int index) {
        is_number_strict(index);
        return to_number<T>(index);
    }

    template <typename T>
    inline T pop_number() {
        const T ret = to_number<T>(-1);
        pop();
        return ret;
    }

    template <typename T>
    inline T pop_number(T defaultValue) {
        const T ret = to_number<T>(-1, defaultValue);
        pop();
        return ret;
    }

    template <typename T>
    inline T pop_number_strict() {
        require(1);
        const T ret = to_number_strict<T>(-1);
        pop();
        return ret;
    }

    template <typename T>
    inline void push_number(T val) {
        lua_pushnumber(State_, static_cast<lua_Number>(val));
    }

    inline bool is_bool(int index) {
        return lua_isboolean(State_, index);
    }

    inline void is_bool_strict(int index) {
        if (!is_bool(index)) {
            ythrow TError() << "internal lua error (not a boolean)";
        }
    }

    inline bool to_bool(int index) {
        return lua_toboolean(State_, index);
    }

    inline bool to_bool(int index, bool defaultValue) {
        return is_bool(index) ? to_bool(index) : defaultValue;
    }

    inline bool to_bool_strict(int index) {
        is_bool_strict(index);
        return to_bool(index);
    }

    inline bool pop_bool() {
        const bool ret = to_bool(-1);
        pop();
        return ret;
    }

    inline bool pop_bool(bool defaultValue) {
        const bool ret = to_bool(-1, defaultValue);
        pop();
        return ret;
    }

    inline bool pop_bool_strict() {
        require(1);
        const bool ret = to_bool_strict(-1);
        pop();
        return ret;
    }

    inline void push_bool(bool val) {
        lua_pushboolean(State_, val);
    }

    inline bool is_nil(int index) {
        return lua_isnil(State_, index);
    }

    inline void is_nil_strict(int index) {
        if (!is_nil(index)) {
            ythrow TError() << "internal lua error (not a nil)";
        }
    }

    inline bool pop_nil() {
        const bool ret = is_nil(-1);
        pop();
        return ret;
    }

    inline void pop_nil_strict() {
        require(1);
        is_nil_strict(-1);
        pop();
    }

    inline void push_nil() {
        lua_pushnil(State_);
    }

    inline bool is_void(int index) {
        return lua_islightuserdata(State_, index);
    }

    inline void is_void_strict(int index) {
        if (!is_void(index)) {
            ythrow TError() << "internal lua error (not a void*)";
        }
    }

    inline void* to_void(int index) {
        return lua_touserdata(State_, index);
    }

    inline void* to_void(int index, void* defaultValue) {
        return is_void(index) ? to_void(index) : defaultValue;
    }

    inline void* to_void_strict(int index) {
        is_void_strict(index);
        return to_void(index);
    }

    inline void* pop_void() {
        void* ret = to_void(-1);
        pop();
        return ret;
    }

    inline void* pop_void(void* defaultValue) {
        void* ret = to_void(-1, defaultValue);
        pop();
        return ret;
    }

    inline void* pop_void_strict() {
        require(1);
        void* ret = to_void_strict(-1);
        pop();
        return ret;
    }

    inline void push_void(void* ptr) {
        lua_pushlightuserdata(State_, ptr);
    }

    template <typename T>
    inline bool is_userdata(int index) {
        return to_userdata<T>(index) != NULL;
    }

    template <typename T>
    inline void is_userdata_strict(int index) {
        to_userdata_strict<T>(index);
    }

    template <typename T>
    inline T* to_userdata(int index) {
        return static_cast<T*>(luaL_testudata(State_, index, T::LUA_METATABLE_NAME));
    }

    template <typename T>
    inline T* to_userdata_strict(int index) {
        T* ret = to_userdata<T>(index);
        if (ret == nullptr) {
            ythrow TError() << "internal error (not a userdata '" << T::LUA_METATABLE_NAME << "')";
        }
        return ret;
    }

    template <typename T>
    inline T pop_userdata_strict() {
        require(1);
        const T ret(*to_userdata_strict<T>(-1));
        pop();
        return ret;
    }

    template <typename T>
    inline T* push_userdata(const T& x) {
        // copy constructor
        return new (new_userdata<T>()) T(x);
    }

    template <typename T, typename... R>
    inline T* push_userdata(const R&... r) {
        return new (new_userdata<T>()) T(r...);
    }

    inline void push_global(const char* name) {
        lua_getglobal(State_, name);
    }

    inline void set_global(const char* name, const char* value) {
        lua_pushstring(State_, value);
        set_global(name);
    }

    inline void set_global(const char* name, const double value) {
        lua_pushnumber(State_, value);
        set_global(name);
    }

    inline void set_global(const char* name) {
        lua_setglobal(State_, name);
    }

    inline void register_function(const char* name, lua_CFunction func) {
        lua_register(State_, name, func);
    }

    inline bool is_table(int index) {
        return lua_istable(State_, index);
    }

    inline void is_table_strict(int index) {
        if (!is_table(index)) {
            ythrow TError() << "internal lua error (not a table)";
        }
    }

    inline void create_table(int narr = 0, int nrec = 0) {
        lua_createtable(State_, narr, nrec);
    }

    inline void set_table(int index) {
        lua_settable(State_, index);
    }

    inline void get_field(int index, const char* key) {
        lua_getfield(State_, index, key);
    }

    inline void set_field(int index, const char* key) {
        lua_setfield(State_, index, key);
    }

    inline void rawseti(int index, int arr_index) {
        lua_rawseti(State_, index, arr_index);
    }

    inline int check_stack(int extra) {
        return lua_checkstack(State_, extra);
    }

    inline void ensure_stack(int extra) {
        if (!check_stack(extra)) {
            ythrow TError() << "cannot allocate more lua stack space";
        };
    }

    inline void require(int n) {
        if (on_stack() < n) {
            ythrow TError() << "lua requirement failed";
        }
    }

    inline void call(int args, int rets) {
        if (lua_pcall(State_, args, rets, 0)) {
            error();
        }
    }

    void call(int args, int rets, TDuration time_limit, int count = 1000);
    void call(int args, int rets, int limit);

    inline void remove(int index) {
        lua_remove(State_, index);
    }

    inline int next(int index) {
        return lua_next(State_, index);
    }

    inline void pop(int n = 1) {
        lua_pop(State_, Min(n, on_stack()));
    }

    inline void push_value(int index) {
        lua_pushvalue(State_, index);
    }

    inline int on_stack() {
        return lua_gettop(State_);
    }

    inline void gc() {
        lua_gc(State_, LUA_GCCOLLECT, 0);
    }

    inline TLuaStateHolder new_thread() {
        return lua_newthread(State_);
    }

    inline bool is_thread(int index) {
        return lua_isthread(State_, index);
    }

    inline void is_thread_strict(int index) {
        if (!is_thread(index)) {
            ythrow TError() << "internal lua error (not a thread)";
        }
    }

    inline TLuaStateHolder to_thread(int index) {
        return lua_tothread(State_, index);
    }

    inline TLuaStateHolder to_thread_strict(int index) {
        is_thread_strict(index);
        return to_thread(index);
    }

    void Load(IInputStream* in, TZtStringBuf name);
    void Dump(IOutputStream* out);
    void DumpStack(IOutputStream* out);

private:
    template <typename T>
    inline void set_metatable() {
        if (luaL_newmetatable(State_, T::LUA_METATABLE_NAME)) {
            // metatable isn't registered yet
            push_string("__index");
            push_value(-2); // pushes the metatable
            set_table(-3);  // metatable.__index = metatable
            luaL_setfuncs(State_, T::LUA_FUNCTIONS, 0);
        }
        lua_setmetatable(State_, -2);
    }

    template <typename T>
    inline void* new_userdata() {
        void* p = lua_newuserdata(State_, sizeof(T));
        set_metatable<T>();
        return p;
    }

private:
    static void* Alloc(void* ud, void* ptr, size_t osize, size_t nsize);
    static void* AllocLimit(void* ud, void* ptr, size_t osize, size_t nsize);

private:
    size_t AllocFree = 0;
    THolder<lua_State, TDeleteState> MyState_;
    lua_State* State_ = nullptr;
};

namespace NLua {
    template <int func(TLuaStateHolder&)>
    int FunctionHandler(lua_State* L) {
        try {
            TLuaStateHolder state(L);
            return func(state);
        } catch (const yexception& e) {
            lua_pushstring(L, e.what());
        }
        return lua_error(L);
    }

    template <class T, int (T::*Method)(TLuaStateHolder&)>
    int MethodHandler(lua_State* L) {
        T* x = static_cast<T*>(luaL_checkudata(L, 1, T::LUA_METATABLE_NAME));
        try {
            TLuaStateHolder state(L);
            return (x->*Method)(state);
        } catch (const yexception& e) {
            lua_pushstring(L, e.what());
        }
        return lua_error(L);
    }

    template <class T, int (T::*Method)(TLuaStateHolder&) const>
    int MethodConstHandler(lua_State* L) {
        const T* x = static_cast<const T*>(luaL_checkudata(L, 1, T::LUA_METATABLE_NAME));
        try {
            TLuaStateHolder state(L);
            return (x->*Method)(state);
        } catch (const yexception& e) {
            lua_pushstring(L, e.what());
        }
        return lua_error(L);
    }

    template <class T>
    int Destructor(lua_State* L) {
        T* x = static_cast<T*>(luaL_checkudata(L, 1, T::LUA_METATABLE_NAME));
        try {
            x->~T();
            return 0;
        } catch (const yexception& e) {
            lua_pushstring(L, e.what());
        }
        return lua_error(L);
    }

    TBuffer& Compile(TStringBuf script, TBuffer& buffer);

    struct TStackDumper {
        TStackDumper(TLuaStateHolder& state)
            : State(state)
        {
        }

        TLuaStateHolder& State;
    };

    struct TMarkedStackDumper: public TStackDumper {
        TMarkedStackDumper(TLuaStateHolder& state, TStringBuf mark)
            : TStackDumper(state)
            , Mark(mark)
        {
        }

        TStringBuf Mark;
    };

    inline TMarkedStackDumper DumpStack(TLuaStateHolder& state, TStringBuf mark) {
        return TMarkedStackDumper(state, mark);
    }

    inline TStackDumper DumpStack(TLuaStateHolder& state) {
        return TStackDumper(state);
    }

}
