#include "stdafx.h"
#include "cpu_affinity.h"

#if defined(__FreeBSD__) && (__FreeBSD__ >= 7)
#include <sys/param.h>
#include <sys/cpuset.h>
#elif defined(_linux_)
#include <pthread.h>
#include <util/stream/file.h>
#include <util/string/printf.h>
#endif

namespace NNetliba {
    class TCPUSet {
    public:
        static constexpr int MAX_SIZE = 128;

    private:
#if defined(__FreeBSD__) && (__FreeBSD__ >= 7)
#define NUMCPU ((CPU_MAXSIZE > MAX_SIZE) ? 1 : (MAX_SIZE / CPU_MAXSIZE))
        cpuset_t CpuInfo[NUMCPU];

    public:
        bool GetAffinity() {
            int error = cpuset_getaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, sizeof(CpuInfo), CpuInfo);
            return error == 0;
        }
        bool SetAffinity() {
            int error = cpuset_setaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, sizeof(CpuInfo), CpuInfo);
            return error == 0;
        }
        bool IsSet(size_t i) {
            return CPU_ISSET(i, CpuInfo);
        }
        void Set(size_t i) {
            CPU_SET(i, CpuInfo);
        }
#elif defined(_linux_)
    public:
#define NUMCPU ((CPU_SETSIZE > MAX_SIZE) ? 1 : (MAX_SIZE / CPU_SETSIZE))
        cpu_set_t CpuInfo[NUMCPU];

    public:
        bool GetAffinity() {
            int error = pthread_getaffinity_np(pthread_self(), sizeof(CpuInfo), CpuInfo);
            return error == 0;
        }
        bool SetAffinity() {
            int error = pthread_setaffinity_np(pthread_self(), sizeof(CpuInfo), CpuInfo);
            return error == 0;
        }
        bool IsSet(size_t i) {
            return CPU_ISSET(i, CpuInfo);
        }
        void Set(size_t i) {
            CPU_SET(i, CpuInfo);
        }
#else
    public:
        bool GetAffinity() {
            return true;
        }
        bool SetAffinity() {
            return true;
        }
        bool IsSet(size_t i) {
            Y_UNUSED(i);
            return true;
        }
        void Set(size_t i) {
            Y_UNUSED(i);
        }
#endif

        TCPUSet() {
            Clear();
        }
        void Clear() {
            memset(this, 0, sizeof(*this));
        }
    };

    static TMutex CPUSetsLock;
    struct TCPUSetInfo {
        TCPUSet CPUSet;
        bool IsOk;

        TCPUSetInfo()
            : IsOk(false)
        {
        }
    };
    static THashMap<int, TCPUSetInfo> CPUSets;

    void BindToSocket(int n) {
        TGuard<TMutex> gg(CPUSetsLock);
        if (CPUSets.find(n) == CPUSets.end()) {
            TCPUSetInfo& res = CPUSets[n];

            bool foundCPU = false;
#ifdef _linux_
            for (int cpuId = 0; cpuId < TCPUSet::MAX_SIZE; ++cpuId) {
                try { // I just wanna check if file exists, I don't want your stinking exceptions :/
                    TIFStream f(Sprintf("/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpuId).c_str());
                    TString s;
                    if (f.ReadLine(s) && !s.empty()) {
                        //printf("cpu%d - %s\n", cpuId, s.c_str());
                        int physCPU = atoi(s.c_str());
                        if (physCPU == 0) {
                            res.IsOk = true;
                            res.CPUSet.Set(cpuId);
                            foundCPU = true;
                        }
                    } else {
                        break;
                    }
                } catch (const TFileError&) {
                    break;
                }
            }
#endif
            if (!foundCPU && n == 0) {
                for (int i = 0; i < 6; ++i) {
                    res.CPUSet.Set(i);
                }
                res.IsOk = true;
                foundCPU = true;
            }
        }
        {
            TCPUSetInfo& cc = CPUSets[n];
            if (cc.IsOk) {
                cc.CPUSet.SetAffinity();
            }
        }
    }

}
