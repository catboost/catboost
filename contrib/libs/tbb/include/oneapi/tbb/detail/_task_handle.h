/*
    Copyright (c) 2020-2025 Intel Corporation
    Copyright (c) 2025 UXL Foundation Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/


#ifndef __TBB_task_handle_H
#define __TBB_task_handle_H

#include "_config.h"
#include "_task.h"
#include "_small_object_pool.h"
#include "_utils.h"
#include <memory>

namespace tbb {
namespace detail {

namespace d1 { class task_group_context; class wait_context; struct execution_data; }
namespace d2 {

class task_handle;

#if __TBB_PREVIEW_TASK_GROUP_EXTENSIONS

class task_handle_task;
class task_dynamic_state;

struct successor_list_node {
    successor_list_node* next_node = nullptr;
    task_dynamic_state* successor_state = nullptr;
    d1::small_object_allocator allocator;

    successor_list_node(task_dynamic_state* state, d1::small_object_allocator& alloc)
        : successor_state(state), allocator(alloc)
    {}

    void destroy() {
        allocator.delete_object(this);
    }
};

class task_dynamic_state {
public:
    task_dynamic_state(task_handle_task* task, d1::small_object_allocator& alloc)
        : m_task(task)
        , m_successor_list_head(nullptr)
        , m_new_completion_point(nullptr)
        , m_num_dependencies(0)
        , m_num_references(1) // reserves a task co-ownership for dynamic state
        , m_allocator(alloc)
    {}

    void reserve() { ++m_num_references; }

    void release() {
        if (--m_num_references == 0) {
            task_dynamic_state* new_completion_point = m_new_completion_point.load(std::memory_order_relaxed);
            // There was a new completion point assigned to the current one by transferring the completion
            // Need to unregister the current dynamic state as a co-owner
            if (new_completion_point) new_completion_point->release();
            m_allocator.delete_object(this);
        }
    }

    void register_dependency() {
        if (m_num_dependencies++ == 0) {
            // Register an additional dependency for a task_handle owning the current task
            ++m_num_dependencies;
        }
    }

    // Returns true if the released dependency was the last remaining one; false otherwise
    bool release_dependency() {
        auto updated_dependency_counter = --m_num_dependencies;
        return updated_dependency_counter == 0;
    }

    bool has_dependencies() const {
        return m_num_dependencies.load(std::memory_order_acquire) != 0;
    }

    task_handle_task* complete_and_try_get_successor();

    void add_successor(task_handle&  successor);
    void add_successor_node(successor_list_node* new_successor_node, successor_list_node* current_successor_list_head);
    void add_successor_list(successor_list_node* successor_list);

    using successor_list_state_flag = std::uintptr_t;
    static constexpr successor_list_state_flag COMPLETED_FLAG = ~std::uintptr_t(0);
    static constexpr successor_list_state_flag TRANSFERRED_FLAG = ~std::uintptr_t(1);

    static bool represents_completed_task(successor_list_node* list_head) {
        return list_head == reinterpret_cast<successor_list_node*>(COMPLETED_FLAG);
    }

    static bool represents_transferred_completion(successor_list_node* list_head) {
        return list_head == reinterpret_cast<successor_list_node*>(TRANSFERRED_FLAG);
    }

    successor_list_node* fetch_successor_list(successor_list_state_flag new_list_state_flag) {
        return m_successor_list_head.exchange(reinterpret_cast<successor_list_node*>(new_list_state_flag));
    }

    void transfer_completion_to(task_dynamic_state* new_completion_point) {
        __TBB_ASSERT(new_completion_point != nullptr, nullptr);
        // Register current dynamic state as a co-owner of the new_completion_point
        // to prevent it's early destruction
        new_completion_point->reserve();
        m_new_completion_point.store(new_completion_point, std::memory_order_relaxed);
        successor_list_node* successor_list = fetch_successor_list(TRANSFERRED_FLAG);
        new_completion_point->add_successor_list(successor_list);
    }

    task_handle_task* get_task() { return m_task; }

private:
    task_handle_task* m_task;
    std::atomic<successor_list_node*> m_successor_list_head;
    std::atomic<task_dynamic_state*> m_new_completion_point;
    std::atomic<std::size_t> m_num_dependencies;
    std::atomic<std::size_t> m_num_references;
    d1::small_object_allocator m_allocator;
};
#endif // __TBB_PREVIEW_TASK_GROUP_EXTENSIONS

class task_handle_task : public d1::task {
    // Pointer to the instantiation of destroy_function_task with the concrete derived type,
    // used for correct destruction and deallocation of the task
    using destroy_func_type = void (*)(task_handle_task*, d1::small_object_allocator&, const d1::execution_data*);

    // Reuses the first std::uint64_t field (previously m_version_and_traits) to maintain backward compatibility
    // The type of the first field remains std::uint64_t to preserve alignment and offset of subsequent member variables.
    static_assert(sizeof(destroy_func_type) <= sizeof(std::uint64_t), "Cannot fit destroy pointer into std::uint64_t");
    std::uint64_t m_destroy_func;

    d1::wait_tree_vertex_interface* m_wait_tree_vertex;
    d1::task_group_context& m_ctx;
    d1::small_object_allocator m_allocator;
#if __TBB_PREVIEW_TASK_GROUP_EXTENSIONS
    std::atomic<task_dynamic_state*> m_dynamic_state;
#endif
public:
    void destroy(const d1::execution_data* ed = nullptr) {
        destroy_func_type destroy_func = reinterpret_cast<destroy_func_type>(m_destroy_func);
        if (destroy_func != nullptr) {
            // If the destroy function is set for the current instantiation - use it
            (*destroy_func)(this, m_allocator, ed);
        } else {
            // Otherwise, the object was compiled with the old version of the library
            // Destroy the object and let the memory leak since the derived type is unknown
            // and the object cannot be deallocated properly
            this->~task_handle_task();
        }
    }

    task_handle_task(d1::wait_tree_vertex_interface* vertex, d1::task_group_context& ctx,
                     d1::small_object_allocator& alloc, destroy_func_type destroy_func)
        : m_destroy_func(reinterpret_cast<std::uint64_t>(destroy_func))
        , m_wait_tree_vertex(vertex)
        , m_ctx(ctx)
        , m_allocator(alloc)
#if __TBB_PREVIEW_TASK_GROUP_EXTENSIONS
        , m_dynamic_state(nullptr)
#endif
    {
        m_wait_tree_vertex->reserve();
    }

    ~task_handle_task() override {
        m_wait_tree_vertex->release();
#if __TBB_PREVIEW_TASK_GROUP_EXTENSIONS
        task_dynamic_state* current_state = m_dynamic_state.load(std::memory_order_relaxed);
        if (current_state != nullptr) {
            current_state->release();
        }
#endif
    }

    d1::task_group_context& ctx() const { return m_ctx; }

#if __TBB_PREVIEW_TASK_GROUP_EXTENSIONS
    // Returns the dynamic state associated with the task. If the state has not been initialized, initializes it.
    task_dynamic_state* get_dynamic_state() {
        task_dynamic_state* current_state = m_dynamic_state.load(std::memory_order_acquire);

        if (current_state == nullptr) {
            d1::small_object_allocator alloc;

            task_dynamic_state* new_state = alloc.new_object<task_dynamic_state>(this, alloc);

            if (m_dynamic_state.compare_exchange_strong(current_state, new_state)) {
                current_state = new_state;
            } else {
                // CAS failed, current_state points to the dynamic state created by another thread
                alloc.delete_object(new_state);
            }
        }

        __TBB_ASSERT(current_state != nullptr, "Failed to create dynamic state");
        return current_state;
    }

    task_handle_task* complete_and_try_get_successor() {
        task_handle_task* next_task = nullptr;

        task_dynamic_state* current_state = m_dynamic_state.load(std::memory_order_relaxed);
        if (current_state != nullptr) {
            next_task = current_state->complete_and_try_get_successor();
        }
        return next_task;
    }

    // Returns true if the released dependency was the last remaining one; false otherwise
    bool release_dependency() {
        task_dynamic_state* current_state = m_dynamic_state.load(std::memory_order_relaxed);
        __TBB_ASSERT(current_state != nullptr && current_state->has_dependencies(),
                     "release_dependency was called for task without dependencies");
        return current_state->release_dependency();
    }

    bool has_dependencies() const {
        task_dynamic_state* current_state = m_dynamic_state.load(std::memory_order_relaxed);
        return current_state ? current_state->has_dependencies() : false;
    }

    void transfer_completion_to(task_handle& receiving_task);
#endif
};

class task_handle {
    struct task_handle_task_deleter {
        void operator()(task_handle_task* p){ p->destroy(); }
    };
    using handle_impl_t = std::unique_ptr<task_handle_task, task_handle_task_deleter>;

    handle_impl_t m_handle = {nullptr};
public:
    task_handle() = default;
    task_handle(task_handle&&) = default;
    task_handle& operator=(task_handle&&) = default;

    explicit operator bool() const noexcept { return static_cast<bool>(m_handle); }

    friend bool operator==(task_handle const& th, std::nullptr_t) noexcept;
    friend bool operator==(std::nullptr_t, task_handle const& th) noexcept;

    friend bool operator!=(task_handle const& th, std::nullptr_t) noexcept;
    friend bool operator!=(std::nullptr_t, task_handle const& th) noexcept;

private:
    friend struct task_handle_accessor;
#if __TBB_PREVIEW_TASK_GROUP_EXTENSIONS
    friend class task_completion_handle;
#endif

    task_handle(task_handle_task* t) : m_handle {t}{}

    d1::task* release() {
       return m_handle.release();
    }
};

struct task_handle_accessor {
    static task_handle construct(task_handle_task* t) { return {t}; }

    static task_handle_task* release(task_handle& th) {
        return th.m_handle.release();
    }

    static d1::task_group_context& ctx_of(task_handle& th) {
        __TBB_ASSERT(th.m_handle, "ctx_of does not expect empty task_handle.");
        return th.m_handle->ctx();
    }

#if __TBB_PREVIEW_TASK_GROUP_EXTENSIONS
    static task_dynamic_state* get_task_dynamic_state(task_handle& th) {
        return th.m_handle->get_dynamic_state();
    }
#endif
};

inline bool operator==(task_handle const& th, std::nullptr_t) noexcept {
    return th.m_handle == nullptr;
}
inline bool operator==(std::nullptr_t, task_handle const& th) noexcept {
    return th.m_handle == nullptr;
}

inline bool operator!=(task_handle const& th, std::nullptr_t) noexcept {
    return th.m_handle != nullptr;
}

inline bool operator!=(std::nullptr_t, task_handle const& th) noexcept {
    return th.m_handle != nullptr;
}

#if __TBB_PREVIEW_TASK_GROUP_EXTENSIONS
inline void task_dynamic_state::add_successor_node(successor_list_node* new_successor_node,
                                                   successor_list_node* current_successor_list_head)
{
    __TBB_ASSERT(new_successor_node != nullptr, nullptr);

    new_successor_node->next_node = current_successor_list_head;

    while (!m_successor_list_head.compare_exchange_strong(current_successor_list_head, new_successor_node)) {
        // Other thread updated the head of the list

        if (represents_completed_task(current_successor_list_head)) {
            // Current task has completed while we tried to insert the successor to the list
            new_successor_node->successor_state->release_dependency();
            new_successor_node->destroy();
            break;
        } else if (represents_transferred_completion(current_successor_list_head)) {
            // Redirect successor to the task received the completion
            task_dynamic_state* new_completion_point = m_new_completion_point.load(std::memory_order_relaxed);
            __TBB_ASSERT(new_completion_point, "successor list is marked as transferred, but new dynamic state is not set");
            new_completion_point->add_successor_node(new_successor_node, new_completion_point->m_successor_list_head.load(std::memory_order_acquire));
            break;
        }

        new_successor_node->next_node = current_successor_list_head;
    }
}

inline void task_dynamic_state::add_successor(task_handle& successor) {
    successor_list_node* current_successor_list_head = m_successor_list_head.load(std::memory_order_acquire);

    if (!represents_completed_task(current_successor_list_head)) {
        if (represents_transferred_completion(current_successor_list_head)) {
            // Redirect successor to the task received the completion
            task_dynamic_state* new_completion_point = m_new_completion_point.load(std::memory_order_relaxed);
            __TBB_ASSERT(new_completion_point, "successor list is marked as transferred, but new dynamic state is not set");
            new_completion_point->add_successor(successor);
        } else {
            task_dynamic_state* successor_state = task_handle_accessor::get_task_dynamic_state(successor);
            successor_state->register_dependency();
    
            d1::small_object_allocator alloc;
            successor_list_node* new_successor_node = alloc.new_object<successor_list_node>(successor_state, alloc);
            add_successor_node(new_successor_node, current_successor_list_head);
        }
    }
}

inline void task_dynamic_state::add_successor_list(successor_list_node* successor_list) {
    if (successor_list == nullptr) return;

    successor_list_node* last_node = successor_list;

    while (last_node->next_node != nullptr) {
        last_node = last_node->next_node;
    }

    successor_list_node* current_successor_list_head = m_successor_list_head.load(std::memory_order_acquire);
    last_node->next_node = current_successor_list_head;

    while (!m_successor_list_head.compare_exchange_strong(current_successor_list_head, successor_list)) {
        __TBB_ASSERT(!represents_completed_task(current_successor_list_head) &&
                     !represents_transferred_completion(current_successor_list_head),
                     "Task receiving the completion was executed or completed");
        // Other thread updated the head of the list
        last_node->next_node = current_successor_list_head;
    }
}

inline task_handle_task* task_dynamic_state::complete_and_try_get_successor() {
    task_handle_task* next_task = nullptr;

    successor_list_node* node = m_successor_list_head.load(std::memory_order_acquire);

    // Doing a single check is enough since the this function is called after the task body and
    // the state of the list cannot change to transferred
    if (!represents_transferred_completion(node)) {
        node = fetch_successor_list(COMPLETED_FLAG);

        while (node != nullptr) {
            task_dynamic_state* successor_state = node->successor_state;

            if (successor_state->release_dependency()) {
                task_handle_task* successor_task = successor_state->get_task();
                if (next_task == nullptr) {
                    next_task = successor_task;
                } else {
                    d1::spawn(*successor_task, successor_task->ctx());
                }
            }

            successor_list_node* next_node = node->next_node;
            node->destroy();
            node = next_node;
        }
    }
    return next_task;
}

inline void task_handle_task::transfer_completion_to(task_handle& receiving_task) {
    __TBB_ASSERT(receiving_task, nullptr);
    task_dynamic_state* current_state = m_dynamic_state.load(std::memory_order_relaxed);
    
    // If dynamic state was not created for currently executing task,
    // it cannot have successors or associated completion handles
    if (current_state != nullptr) {
        current_state->transfer_completion_to(task_handle_accessor::get_task_dynamic_state(receiving_task));
    }
}

class task_completion_handle {
public:
    task_completion_handle() : m_task_state(nullptr) {}

    task_completion_handle(const task_completion_handle& other) 
        : m_task_state(other.m_task_state)
    {
        // Register one more co-owner of the dynamic state
        if (m_task_state) m_task_state->reserve();
    }
    task_completion_handle(task_completion_handle&& other)
        : m_task_state(other.m_task_state)
    {
        other.m_task_state = nullptr;
    }

    task_completion_handle(const task_handle& th)
        : m_task_state(nullptr)
    {
        __TBB_ASSERT(th, "Construction of task_completion_handle from an empty task_handle");
        m_task_state = th.m_handle->get_dynamic_state();
        // Register one more co-owner of the dynamic state
        m_task_state->reserve();
    }

    ~task_completion_handle() {
        if (m_task_state) m_task_state->release();
    }

    task_completion_handle& operator=(const task_completion_handle& other) {
        if (m_task_state != other.m_task_state) {
            // Release co-ownership on the previously tracked dynamic state
            if (m_task_state) m_task_state->release();

            m_task_state = other.m_task_state;

            // Register new co-owner of the new dynamic state
            if (m_task_state) m_task_state->reserve();
        }
        return *this;
    }

    task_completion_handle& operator=(task_completion_handle&& other) {
        if (this != &other) {
            // Release co-ownership on the previously tracked dynamic state
            if (m_task_state) m_task_state->release();
    
            m_task_state = other.m_task_state;
            other.m_task_state = nullptr;
        }
        return *this;
    }

    task_completion_handle& operator=(const task_handle& th) {
        __TBB_ASSERT(th, "Assignment of task_completion_state from an empty task_handle");
        task_dynamic_state* th_state = th.m_handle->get_dynamic_state();
        __TBB_ASSERT(th_state != nullptr, "No state in the non-empty task_handle");
        if (m_task_state != th_state) {
            // Release co-ownership on the previously tracked dynamic state
            if (m_task_state) m_task_state->release();

            m_task_state = th_state;

            // Reserve co-ownership on the new dynamic state
            m_task_state->reserve();
        }
        return *this;
    }

    explicit operator bool() const noexcept { return m_task_state != nullptr; }
private:
    friend bool operator==(const task_completion_handle& t, std::nullptr_t) noexcept {
        return t.m_task_state == nullptr;
    }

    friend bool operator==(const task_completion_handle& lhs, const task_completion_handle& rhs) noexcept {
        return lhs.m_task_state == rhs.m_task_state;
    }

#if !__TBB_CPP20_COMPARISONS_PRESENT
    friend bool operator==(std::nullptr_t, const task_completion_handle& t) noexcept {
        return t == nullptr;
    }

    friend bool operator!=(const task_completion_handle& t, std::nullptr_t) noexcept {
        return !(t == nullptr);
    }

    friend bool operator!=(std::nullptr_t, const task_completion_handle& t) noexcept {
        return !(t == nullptr);
    }

    friend bool operator!=(const task_completion_handle& lhs, const task_completion_handle& rhs) noexcept {
        return !(lhs == rhs);
    }
#endif // !__TBB_CPP20_COMPARISONS_PRESENT

    friend struct task_completion_handle_accessor;

    task_dynamic_state* m_task_state;
};

struct task_completion_handle_accessor {
    static task_dynamic_state* get_task_dynamic_state(task_completion_handle& tracker) {
        return tracker.m_task_state;
    }
};
#endif

} // namespace d2
} // namespace detail
} // namespace tbb

#endif /* __TBB_task_handle_H */
