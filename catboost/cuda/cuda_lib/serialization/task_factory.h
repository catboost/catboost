#pragma once

#include <catboost/cuda/cuda_lib/task.h>
#include <library/cpp/object_factory/object_factory.h>
#include <util/generic/buffer.h>
#include <util/generic/hash.h>
#include <util/stream/buffer.h>
#include <util/system/type_name.h>
#include <typeindex>

namespace NCudaLib {
    using TTaskFactory = NObjectFactory::TParametrizedObjectFactory<ICommand, ui64>;

    class TTaskUniqueIdsProvider {
    public:
        template <class TTask>
        void RegisterUniqueId(ui64 taskId) {
            std::type_index index(typeid(TTask));

            if (CommandIds.contains(index)) {
                ythrow TCatBoostException() << "Error: class " << index.name() << " already registered with id " << taskId;
            } else {
                //registers are done on initialization, so this check would not be bottleneck
                for (const auto& entry : CommandIds) {
                    if (entry.second == taskId) {
                        ythrow TCatBoostException() << "Error: Can't register class " << index.name() << ". TaskId " << taskId << " already registered for class " << entry.first.name();
                    }
                }
                CommandIds[index] = taskId;
            }
        }

        template <class TTask>
        ui32 GetUniqueId(const TTask& task) const {
            std::type_index index(typeid(task));
            auto commandId = CommandIds.find(index);
            if (commandId != CommandIds.end()) {
                return commandId->second;
            } else {
                ythrow TCatBoostException() << "Task " << index.name() << " is not registered";
            }
        }

    private:
        THashMap<std::type_index, ui64> CommandIds;
    };

    inline TTaskUniqueIdsProvider& GetTaskUniqueIdsProvider() {
        return *Singleton<TTaskUniqueIdsProvider>();
    }

    template <class TCommand>
    inline void RegisterCommand(ui64 id) {
        GetTaskUniqueIdsProvider().RegisterUniqueId<TCommand>(id);
        TTaskFactory::TRegistrator<TCommand> registrator(id);
    }

    template <class TTask>
    class TTaskRegistrator {
    public:
        explicit TTaskRegistrator(ui64 id) {
            GetTaskUniqueIdsProvider().RegisterUniqueId<TTask>(id);
            TTaskFactory::TRegistrator<TTask> registrator(id);
        }
    };

#define REGISTER_TASK(id, className) \
    static TTaskRegistrator<className> taskRegistrator##id(id);

#define REGISTER_TASK_TEMPLATE(id, className, T) \
    static TTaskRegistrator<className<T>> taskRegistrator##id(id);

#define REGISTER_TASK_TEMPLATE_2(id, className, T1, T2) \
    static TTaskRegistrator<className<T1, T2>> taskRegistrator##id(id);

    using TSerializedTask = TBuffer;

    class TTaskSerializer {
    public:
        static inline THolder<ICommand> LoadCommand(IInputStream* input) {
            ui32 id = 0;
            ::Load(input, id);
            THolder<ICommand> command = THolder<ICommand>(TTaskFactory::Construct(id));
            CB_ENSURE(command, "Error: Can't find object with id " << id);
            command->Load(input);
            return command;
        };

        static inline THolder<ICommand> LoadCommand(const TSerializedTask& task) {
            TBufferInput in(task);
            return LoadCommand(&in);
        }

        template <class TCommand,
                  class TOut>
        static inline void SaveCommand(const TCommand& command, TOut* out) {
            auto& uidsProvider = GetTaskUniqueIdsProvider();
            ui32 key = uidsProvider.GetUniqueId(command);
#if defined(NDEBUG)
            CB_ENSURE(TTaskFactory::Has(key), "Error: no ptr found for class " << TypeName<TCommand>());
#endif
            ::Save(out, key);
            ::Save(out, command);
        };

        template <class TCommand>
        static inline TSerializedTask Serialize(const TCommand& command) {
            TSerializedTask task;
            TBufferOutput out(task);
            SaveCommand(command, &out);
            return task;
        };
    };

}
