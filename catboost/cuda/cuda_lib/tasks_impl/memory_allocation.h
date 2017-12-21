#pragma once

namespace NCudaLib {
    template <EPtrType PtrType>
    class TCudaMallocTask: public IAllocateMemoryTask {
    private:
        ui64 Handle;
        ui64 Size;

    public:
        TCudaMallocTask(ui64 handle,
                        ui64 size)
            : Handle(handle)
            , Size(size)
        {
        }

        ui64 GetHandle() const override {
            return Handle;
        }

        ui64 GetSize() const override {
            return Size;
        }

        EPtrType GetPtrType() const {
            return PtrType;
        }

        Y_SAVELOAD_DEFINE(Handle, Size);
    };

    template <class T>
    class TResetRemotePointerCommand: public IFreeMemoryTask {
    private:
        THandleBasedPointer<T> Ptr;

    public:
        explicit TResetRemotePointerCommand(THandleBasedPointer<T> ptr)
            : Ptr(ptr)
        {
        }

        void Exec() override {
            Ptr.Reset();
        }
    };

    template <class T>
    struct TCreateObjectCommandTrait {
        class TWithoutConstructCommand: public IHostTask {
        private:
            ui64 Handle;

        public:
            explicit TWithoutConstructCommand(ui64 handle)
                : Handle(handle)
            {
            }

            void Exec() override {
                THandleBasedPointer<T> ptr(Handle);
                ptr.Reset(new T);
            }

            Y_SAVELOAD_DEFINE(Handle);
        };

        static THolder<IHostTask> Create(ui64 handle) {
            return new TWithoutConstructCommand(handle);
        }
    };

}
