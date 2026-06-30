/*
*  Copyright 2021 NVIDIA Corporation
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*/

#pragma once
#pragma clang system_header


#include <cub/util_namespace.cuh>
#include <cub/util_device.cuh>


CUB_NAMESPACE_BEGIN


namespace detail
{


namespace temporary_storage
{


class slot;

template <typename T>
class alias;

template <int SlotsCount>
class layout;


/**
 * @brief Temporary storage slot that can be considered a C++ union with an
 *        arbitrary fields count.
 *
 * @warning slot lifetime is defined by the lifetime of the associated layout.
 *          It's impossible to request new array if layout is already mapped.
 *
 * @par A Simple Example
 * @code
 * auto slot = temporary_storage.get_slot(0);
 *
 * // Add fields into the slot
 * // Create an int alias with 0 elements:
 * auto int_array = slot->create_alias<int>();
 * // Create a double alias with 2 elements:
 * auto double_array = slot->create_alias<double>(2);
 * // Create a char alias with 0 elements:
 * auto empty_array = slot->create_alias<char>();
 * // Slot size is defined by double_array size (2 * sizeof(double))
 *
 * if (condition)
 * {
 *   int_array.grow(42);
 *   // Now slot size is defined by int_array size (42 * sizeof(int))
 * }
 *
 * // Temporary storage mapping
 * // ...

 * int *d_int_array = int_array.get();
 * double *d_double_array = double_array.get();
 * char *d_empty_array = empty_array.get(); // Guaranteed to return nullptr
 * @endcode
 */
class slot
{
  std::size_t m_size{};
  void *m_pointer{};

public:
  slot() = default;

  /**
   * @brief Returns an array of type @p T and length @p elements
   */
  template <typename T>
  __host__ __device__ alias<T> create_alias(std::size_t elements = 0);

private:
  __host__ __device__ void set_bytes_required(std::size_t new_size)
  {
    m_size = (max)(m_size, new_size);
  }

  __host__ __device__ std::size_t get_bytes_required() const
  {
    return m_size;
  }

  __host__ __device__ void set_storage(void *ptr) { m_pointer = ptr; }
  __host__ __device__ void *get_storage() const { return m_pointer; }

  template <typename T>
  friend class alias;

  template <int>
  friend class layout;
};


/**
 * @brief Named memory region of a temporary storage slot
 *
 * @par Overview
 * This class provides a typed wrapper of a temporary slot memory region.
 * It can be considered as a field in the C++ union. It's only possible to
 * increase the array size.
 *
 * @warning alias lifetime is defined by the lifetime of the associated slot
 *          It's impossible to grow the array if the layout is already mapped.
 */
template <typename T>
class alias
{
  slot &m_slot;
  std::size_t m_elements{};

  __host__ __device__ explicit alias(slot &slot,
                                     std::size_t elements = 0)
    : m_slot(slot)
    , m_elements(elements)
  {
    this->update_slot();
  }

  __host__ __device__ void update_slot()
  {
    m_slot.set_bytes_required(m_elements * sizeof(T));
  }

public:
  alias() = delete;

  /**
   * @brief Increases the number of elements
   *
   * @warning
   *   This method should be called before temporary storage mapping stage.
   *
   * @param[in] new_elements Increases the memory region occupied in the
   *                         temporary slot to fit up to @p new_elements items
   *                         of type @p T.
   */
  __host__ __device__ void grow(std::size_t new_elements)
  {
    m_elements = new_elements;
    this->update_slot();
  }

  /**
   * @brief Returns pointer to array
   *
   * If the @p elements number is equal to zero, or storage layout isn't mapped,
   * @p nullptr is returned.
   */
  __host__ __device__ T *get() const
  {
    if (m_elements == 0)
    {
      return nullptr;
    }

    return reinterpret_cast<T *>(m_slot.get_storage());
  }

  friend class slot;
};


template <typename T>
__host__ __device__ alias<T> slot::create_alias(std::size_t elements)
{
  return alias<T>(*this, elements);
}


/**
 * @brief Temporary storage layout represents a structure with
 *        @p SlotsCount union-like fields
 *
 * The layout can be mapped to a temporary buffer only once.
 *
 * @par A Simple Example
 * @code
 * cub::detail::temporary_storage::layout<3> temporary_storage;
 *
 * auto slot_1 = temporary_storage.get_slot(0);
 * auto slot_2 = temporary_storage.get_slot(1);
 *
 * // Add fields into the first slot
 * auto int_array = slot_1->create_alias<int>(1);
 * auto double_array = slot_1->create_alias<double>(2);
 *
 * // Add fields into the second slot
 * auto char_array = slot_2->create_alias<char>();
 *
 * // The equivalent C++ structure could look like
 * // struct StorageLayout
 * // {
 * //   union {
 * //   } slot_0;
 * //   std::byte padding_0[256 - sizeof (slot_0)];
 * //
 * //   union {
 * //     int alias_0[1];
 * //     double alias_1[2];
 * //   } slot_1;
 * //   std::byte padding_1[256 - sizeof (slot_1)];
 * //
 * //   union {
 * //     char alias_0[0];
 * //   } slot_2;
 * //   std::byte padding_2[256 - sizeof (slot_2)];
 * // };
 *
 * // The third slot is empty
 *
 * // Temporary storage mapping
 * if (d_temp_storage == nullptr)
 * {
 *   temp_storage_bytes = temporary_storage.get_size();
 *   return;
 * }
 * else
 * {
 *   temporary_storage.map_to_buffer(d_temp_storage, temp_storage_bytes);
 * }
 *
 * // Use pointers
 * int *d_int_array = int_array.get();
 * double *d_double_array = double_array.get();
 * char *d_char_array = char_array.get();
 * @endcode
 */
template <int SlotsCount>
class layout
{
  slot m_slots[SlotsCount];
  std::size_t m_sizes[SlotsCount];
  void *m_pointers[SlotsCount];
  bool m_layout_was_mapped {};

public:
  layout() = default;

  __host__ __device__ slot *get_slot(int slot_id)
  {
    if (slot_id < SlotsCount)
    {
      return &m_slots[slot_id];
    }

    return nullptr;
  }

  /**
   * @brief Returns required temporary storage size in bytes
   */
  __host__ __device__ std::size_t get_size()
  {
    this->prepare_interface();

    // AliasTemporaries can return error only in mapping stage,
    // so it's safe to ignore it here.
    std::size_t temp_storage_bytes{};
    AliasTemporaries(nullptr, temp_storage_bytes, m_pointers, m_sizes);

    if (temp_storage_bytes == 0)
    {
      // The current CUB convention implies that there are two stages for each
      // device-scope function call. The first one returns the required storage
      // size. The second stage consumes temporary storage to perform some work.
      // The only way to distinguish between the two stages is by checking the
      // value of the temporary storage pointer. If zero bytes are requested,
      // `cudaMalloc` will return `nullptr`. This fact makes it impossible to
      // distinguish between the two stages, so we request some fixed amount of
      // bytes (even if we don't need it) to have a non-null temporary storage
      // pointer.
      return 1;
    }

    return temp_storage_bytes;
  }

  /**
   * @brief Maps the layout to the temporary storage buffer.
   */
  __host__ __device__ cudaError_t map_to_buffer(void *d_temp_storage,
                                                std::size_t temp_storage_bytes)
  {
    if (m_layout_was_mapped)
    {
      return cudaErrorAlreadyMapped;
    }

    this->prepare_interface();

    cudaError_t error = cudaSuccess;
    if ((error = AliasTemporaries(d_temp_storage,
                                  temp_storage_bytes,
                                  m_pointers,
                                  m_sizes)))
    {
      return error;
    }

    for (std::size_t slot_id = 0; slot_id < SlotsCount; slot_id++)
    {
      m_slots[slot_id].set_storage(m_pointers[slot_id]);
    }

    m_layout_was_mapped = true;
    return error;
  }

private:
  __host__ __device__ void prepare_interface()
  {
    if (m_layout_was_mapped)
    {
      return;
    }

    for (std::size_t slot_id = 0; slot_id < SlotsCount; slot_id++)
    {
      const std::size_t slot_size = m_slots[slot_id].get_bytes_required();

      m_sizes[slot_id]    = slot_size;
      m_pointers[slot_id] = nullptr;
    }
  }
};

} // namespace temporary_storage

} // namespace detail

CUB_NAMESPACE_END
