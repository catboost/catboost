#pragma once

#include <string.h>

#include <util/generic/ptr.h>
#include <util/generic/singleton.h>
#include <util/generic/string.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <util/thread/lfqueue.h>

#include "http_common.h"

namespace NNeh {
    namespace NHttp2 {
        // TConn must be refcounted and contain methods:
        //  void SetCached(bool) noexcept
        //  bool IsValid() const noexcept
        //  void Close() noexcept
        template <class TConn>
        class TConnCache {
            struct TCounter : TAtomicCounter {
                inline void IncCount(const TConn* const&) {
                    Inc();
                }

                inline void DecCount(const TConn* const&) {
                    Dec();
                }
            };

        public:
            typedef TIntrusivePtr<TConn> TConnRef;

            class TConnList: public TLockFreeQueue<TConn*, TCounter> {
            public:
                ~TConnList() {
                    Clear();
                }

                inline void Clear() {
                    TConn* conn;

                    while (this->Dequeue(&conn)) {
                        conn->Close();
                        conn->UnRef();
                    }
                }

                inline size_t Size() {
                    return this->GetCounter().Val();
                }
            };

            inline void Put(TConnRef& conn, size_t addrId) {
                conn->SetCached(true);
                ConnList(addrId).Enqueue(conn.Get());
                conn->Ref();
                Y_UNUSED(conn.Release());
                CachedConn_.Inc();
            }

            bool Get(TConnRef& conn, size_t addrId) {
                TConnList& connList = ConnList(addrId);
                TConn* connTmp;

                while (connList.Dequeue(&connTmp)) {
                    connTmp->SetCached(false);
                    CachedConn_.Dec();
                    if (connTmp->IsValid()) {
                        TConnRef(connTmp).Swap(conn);
                        connTmp->DecRef();

                        return true;
                    } else {
                        connTmp->UnRef();
                    }
                }
                return false;
            }

            inline size_t Size() const noexcept {
                return CachedConn_.Val();
            }

            inline size_t Validate(size_t addrId) {
                TConnList& cl = Lst_.Get(addrId);
                return Validate(cl);
            }

            //close/remove part of the connections from cache
            size_t Purge(size_t addrId, size_t frac256) {
                TConnList& cl = Lst_.Get(addrId);
                size_t qsize = cl.Size();
                if (!qsize) {
                    return 0;
                }

                size_t purgeCounter = ((qsize * frac256) >> 8);
                if (!purgeCounter && qsize >= 2) {
                    purgeCounter = 1;
                }

                size_t pc = 0;
                {
                    TConn* conn;
                    while (purgeCounter-- && cl.Dequeue(&conn)) {
                        conn->SetCached(false);
                        if (conn->IsValid()) {
                            conn->Close();
                        }
                        CachedConn_.Dec();
                        conn->UnRef();
                        ++pc;
                    }
                }
                pc += Validate(cl);

                return pc;
            }

        private:
            inline TConnList& ConnList(size_t addrId) {
                return Lst_.Get(addrId);
            }

            inline size_t Validate(TConnList& cl) {
                size_t pc = 0;
                size_t nc = cl.Size();

                TConn* conn;
                while (nc-- && cl.Dequeue(&conn)) {
                    if (conn->IsValid()) {
                        cl.Enqueue(conn);
                    } else {
                        ++pc;
                        conn->SetCached(false);
                        CachedConn_.Dec();
                        conn->UnRef();
                    }
                }

                return pc;
            }

            NNeh::NHttp::TLockFreeSequence<TConnList> Lst_;
            TAtomicCounter CachedConn_;
        };
    }
}
