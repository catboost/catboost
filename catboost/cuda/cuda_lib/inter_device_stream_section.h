#pragma once

#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/cuda_lib/cuda_events_provider.h>
#include <library/cpp/deprecated/atomic/atomic.h>
#include <util/system/spinlock.h>
#include <util/generic/hash.h>

/*
 * Ability to run  code sections, which should be entered and leaved simultaniously on multiple devices (like memcpy  between different devices)
 * CudaStreams are only device-local, but we need also host-local streams
 */
namespace NCudaLib {
    struct TStreamSectionConfig {
        ui64 StreamSectionUid = -1;
        ui64 StreamSectionSize = 0;
        bool LocalOnly = true;

        Y_SAVELOAD_DEFINE(StreamSectionUid, StreamSectionSize, LocalOnly);
    };

    class TStreamSectionProvider {
    private:
        struct TStreamSectionState: public TNonCopyable {
            ui32 Created = 0;
            ui32 Destroyed = 0;
            ui32 Size = 0;
            TAtomic NotReadyToEnter = 0;
            TAtomic NotReadyToLeave = 0;

            explicit TStreamSectionState(ui64 size)
                : Size(size)
                , NotReadyToEnter(size)
                , NotReadyToLeave(size)
            {
            }
        };

        THashMap<ui64, THolder<TStreamSectionState>> Current;
        TAdaptiveLock Lock;
        TAtomic Counter = 0;

        void Leave(ui64 handle) {
            with_lock (Lock) {
                Y_ASSERT(Current.contains(handle));

                auto& state = Current[handle];
                CB_ENSURE(state->Created == state->Size);
                state->Destroyed++;

                if (state->Destroyed == state->Size) {
                    Current.erase(handle);
                }
            }
        }

    public:
        class TStreamSection: public TNonCopyable {
        public:
            bool TryEnter() {
                if (State == EState::Entered) {
                    return NotReadyToEnter == 0;
                }

                if (Event == nullptr) {
                    Y_ASSERT(State == EState::Uninitialized);
                    Event = CreateCudaEvent();
                    Event->Record(Owner);
                    State = EState::Entering;
                }

                if (Event->IsComplete()) {
                    auto notReadyToEnter = AtomicDecrement(NotReadyToEnter);
                    State = EState::Entered;
                    return notReadyToEnter == 0;
                } else {
                    return false;
                }
            }

            bool TryLeave() {
                CB_ENSURE(State == EState::Entered || State == EState::Leaving || State == EState::Left, "Enter section first");
                CB_ENSURE(Event);

                if (State == EState::Left) {
                    return NotReadyToLeave == 0;
                }

                if (State == EState::Entered) {
                    Event->Record(Owner);
                    State = EState::Leaving;
                }

                if (Event->IsComplete()) {
                    auto notReadyToLeave = AtomicDecrement(NotReadyToLeave);
                    State = EState::Left;
                    return notReadyToLeave == 0;
                } else {
                    return false;
                }
            }

            ~TStreamSection() {
                Provider.Leave(Handle);
            }

        private:
            friend class TStreamSectionProvider;

            TStreamSection(ui64 handle,
                           TAtomic& entered,
                           TAtomic& leaved,
                           const TCudaStream& owner,
                           TStreamSectionProvider& provider)
                : Handle(handle)
                , NotReadyToEnter(entered)
                , NotReadyToLeave(leaved)
                , Owner(owner)
                , Provider(provider)
            {
            }

        private:
            enum class EState {
                Uninitialized,
                Entering,
                Entered,
                Leaving,
                Left
            } State = EState::Uninitialized;

            ui64 Handle;
            TAtomic& NotReadyToEnter;
            TAtomic& NotReadyToLeave;
            const TCudaStream& Owner;
            TStreamSectionProvider& Provider;
            TCudaEventPtr Event;
        };

        THolder<TStreamSection> Create(const TStreamSectionConfig& config,
                                       const TCudaStream& stream) {
            return Create(config.StreamSectionUid, config.StreamSectionSize, stream);
        }

        THolder<TStreamSection> Create(ui64 handle,
                                       ui64 size,
                                       const TCudaStream& stream) {
            TStreamSectionState* state;
            with_lock (Lock) {
                if (!Current.contains(handle)) {
                    Current[handle] = MakeHolder<TStreamSectionState>(size);
                    Current[handle]->Created = 1;
                } else {
                    CB_ENSURE(Current[handle]->Size == size);
                    Current[handle]->Created++;
                }
                state = Current[handle].Get();
            }
            return THolder<TStreamSection>(new TStreamSection(handle,
                                      state->NotReadyToEnter,
                                      state->NotReadyToLeave,
                                      stream,
                                      *this));
        }

        ui64 NextUid() {
            return static_cast<ui64>(AtomicIncrement(Counter));
        }
    };

    using TStreamSection = TStreamSectionProvider::TStreamSection;

    inline TStreamSectionProvider& GetStreamSectionProvider() {
        return *Singleton<TStreamSectionProvider>();
    }
}

Y_DECLARE_PODTYPE(NCudaLib::TStreamSectionConfig);
