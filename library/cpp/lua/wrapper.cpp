#include "wrapper.h"

#include <util/datetime/cputimer.h>
#include <util/stream/buffered.h>
#include <util/stream/buffer.h>
#include <util/stream/format.h>
#include <util/stream/input.h>
#include <util/stream/mem.h>
#include <util/stream/output.h>
#include <util/system/sys_alloc.h>

namespace {
    class TLuaCountLimit {
    public:
        TLuaCountLimit(lua_State* state, int count)
            : State(state)
        {
            lua_sethook(State, LuaHookCallback, LUA_MASKCOUNT, count);
        }

        ~TLuaCountLimit() {
            lua_sethook(State, LuaHookCallback, 0, 0);
        }

        static void LuaHookCallback(lua_State* L, lua_Debug*) {
            luaL_error(L, "Lua instruction count limit exceeded");
        }

    private:
        lua_State* State;
    }; // class TLuaCountLimit

    class TLuaTimeLimit {
    public:
        TLuaTimeLimit(lua_State* state, TDuration limit, int count)
            : State(state)
            , Limit(limit)
        {
            lua_pushlightuserdata(State, (void*)LuaHookCallback); //key
            lua_pushlightuserdata(State, (void*)this);            //value
            lua_settable(State, LUA_REGISTRYINDEX);

            lua_sethook(State, LuaHookCallback, LUA_MASKCOUNT, count);
        }

        ~TLuaTimeLimit() {
            lua_sethook(State, LuaHookCallback, 0, 0);
        }

        bool Exceeded() {
            return Timer.Get() > Limit;
        }

        static void LuaHookCallback(lua_State* L, lua_Debug*) {
            lua_pushlightuserdata(L, (void*)LuaHookCallback);
            lua_gettable(L, LUA_REGISTRYINDEX);
            TLuaTimeLimit* t = static_cast<TLuaTimeLimit*>(lua_touserdata(L, -1));
            lua_pop(L, 1);
            if (t->Exceeded()) {
                luaL_error(L, "time limit exceeded");
            }
        }

    private:
        lua_State* State;
        const TDuration Limit;
        TSimpleTimer Timer;
    }; // class TLuaTimeLimit

    class TLuaReader {
    public:
        TLuaReader(IZeroCopyInput* in)
            : In_(in)
        {
        }

        inline void Load(lua_State* state, const char* name) {
            if (lua_load(state, ReadCallback, this, name
#if LUA_VERSION_NUM > 501
                         ,
                         nullptr
#endif
                         ))
            {
                ythrow TLuaStateHolder::TError() << "can not parse lua chunk " << name << ": " << lua_tostring(state, -1);
            }
        }

        static const char* ReadCallback(lua_State*, void* data, size_t* size) {
            return ((TLuaReader*)(data))->Read(size);
        }

    private:
        inline const char* Read(size_t* readed) {
            const char* ret;

            if (*readed = In_->Next(&ret)) {
                return ret;
            }

            return nullptr;
        }

    private:
        IZeroCopyInput* In_;
    }; // class TLuaReader

    class TLuaWriter {
    public:
        TLuaWriter(IOutputStream* out)
            : Out_(out)
        {
        }

        inline void Dump(lua_State* state) {
            if (lua_dump(state, WriteCallback, this)) {
                ythrow TLuaStateHolder::TError() << "can not dump lua state: " << lua_tostring(state, -1);
            }
        }

        static int WriteCallback(lua_State*, const void* data, size_t size, void* user) {
            return ((TLuaWriter*)(user))->Write(data, size);
        }

    private:
        inline int Write(const void* data, size_t size) {
            Out_->Write(static_cast<const char*>(data), size);
            return 0;
        }

    private:
        IOutputStream* Out_;
    }; // class TLuaWriter

} //namespace

void TLuaStateHolder::Load(IInputStream* in, TZtStringBuf name) {
    TBufferedInput wi(in, 8192);
    return TLuaReader(&wi).Load(State_, name.c_str());
}

void TLuaStateHolder::Dump(IOutputStream* out) {
    return TLuaWriter(out).Dump(State_);
}

void TLuaStateHolder::DumpStack(IOutputStream* out) {
    for (int i = lua_gettop(State_) * -1; i < 0; ++i) {
        *out << i << " is " << lua_typename(State_, lua_type(State_, i)) << " (";
        if (is_number(i)) {
            *out << to_number<long long>(i);
        } else if (is_string(i)) {
            *out << to_string(i);
        } else {
            *out << Hex((uintptr_t)lua_topointer(State_, i), HF_ADDX);
        }
        *out << ')' << Endl;
    }
}

void* TLuaStateHolder::Alloc(void* ud, void* ptr, size_t /*osize*/, size_t nsize) {
    (void)ud;

    if (nsize == 0) {
        y_deallocate(ptr);

        return nullptr;
    }

    return y_reallocate(ptr, nsize);
}

void* TLuaStateHolder::AllocLimit(void* ud, void* ptr, size_t osize, size_t nsize) {
    TLuaStateHolder& state = *static_cast<TLuaStateHolder*>(ud);

    if (nsize == 0) {
        y_deallocate(ptr);
        state.AllocFree += osize;

        return nullptr;
    }

    if (state.AllocFree + osize < nsize) {
        return nullptr;
    }

    ptr = y_reallocate(ptr, nsize);

    if (ptr) {
        state.AllocFree += osize;
        state.AllocFree -= nsize;
    }

    return ptr;
}

void TLuaStateHolder::call(int args, int rets, int count) {
    TLuaCountLimit limit(State_, count);
    return call(args, rets);
}

void TLuaStateHolder::call(int args, int rets, TDuration time_limit, int count) {
    TLuaTimeLimit limit(State_, time_limit, count);
    return call(args, rets);
}

template <>
void Out<NLua::TStackDumper>(IOutputStream& out, const NLua::TStackDumper& sd) {
    sd.State.DumpStack(&out);
}

template <>
void Out<NLua::TMarkedStackDumper>(IOutputStream& out, const NLua::TMarkedStackDumper& sd) {
    out << sd.Mark << Endl;
    sd.State.DumpStack(&out);
    out << sd.Mark << Endl;
}

namespace NLua {
    TBuffer& Compile(TStringBuf script, TBuffer& buffer) {
        TMemoryInput input(script.data(), script.size());
        TLuaStateHolder state;
        state.Load(&input, "main");

        TBufferOutput out(buffer);
        state.Dump(&out);
        return buffer;
    }

}
