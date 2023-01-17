#pragma once

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/strbuf.h>
#include <array>
#include <functional>

namespace NEnumSerializationRuntime {
    enum class ESortOrder: int {
        Unordered = 0,         // беспорядок
        Ascending = 1,         // можно искать бинарным поиском, но есть эквивалентные ключи. Гланый ключ находится через lower_bound
        StrictlyAscending = 2, //   + все ключи уникальны
        DirectMapping = 3,     // последовательность целых чисел без пропусков, индекс элемента можно вычислить из его значения не делая поиск
    };

    template <typename TEnumRepresentationType>
    struct TEnumStringPair {
        TEnumRepresentationType Key;
        TStringBuf Name;
    };

    template <typename TEnumRepresentationType>
    constexpr ESortOrder GetKeyFieldSortOrder(const TArrayRef<const TEnumStringPair<TEnumRepresentationType>> initializer) {
        if (initializer.empty()) {
            return ESortOrder::DirectMapping;
        }
        bool direct = true;
        bool strict = true;
        bool sorted = true;
        const auto* data = initializer.data();
        const size_t size = initializer.size();
        TEnumRepresentationType expected = data[0].Key;
        for (size_t i = 1; i < size; ++i) {
            const auto& prev = data[i - 1].Key;
            const auto& next = data[i - 0].Key;
            if (++expected != next) {
                direct = false;
            }
            if (prev >= next) {
                strict = false;
            }
            if (prev > next) {
                sorted = false;
                break;
            }
        }
        return direct   ? ESortOrder::DirectMapping
               : strict ? ESortOrder::StrictlyAscending
               : sorted ? ESortOrder::Ascending
                        : ESortOrder::Unordered;
    }

    template <typename TEnumRepresentationType>
    constexpr ESortOrder GetNameFieldSortOrder(const TArrayRef<const TEnumStringPair<TEnumRepresentationType>> initializer) {
        if (initializer.empty()) {
            return ESortOrder::DirectMapping;
        }
        bool strict = true;
        bool sorted = true;
        const auto* data = initializer.data();
        const size_t size = initializer.size();
        for (size_t i = 1; i < size; ++i) {
            const std::string_view& prev = data[i - 1].Name;
            const std::string_view& next = data[i - 0].Name;
            const int cmp = prev.compare(next);
            if (cmp >= 0) {
                strict = false;
            }
            if (cmp > 0) {
                sorted = false;
                break;
            }
        }
        return strict   ? ESortOrder::StrictlyAscending
               : sorted ? ESortOrder::Ascending
                        : ESortOrder::Unordered;
    }

#if defined(__cpp_lib_array_constexpr) && defined(__cpp_lib_constexpr_algorithms) && defined(__cpp_lib_constexpr_functional)

    // Функция должна состоять из единственного вызова
    // std::stable_sort(v.begin(), v.end(), [](const T& a, const T& b) { return a.Key < b.Key; });
    // и возврата отсортированного массива.
    // Но в C++20 stable_sort ещё не имеет спецификатора constexpr и не может использоваться тут.
    // Поэтому в текущей реализации вместо этого делается обычная нестабильная сортировка пар {ключ элемента, положение элемента}.
    template <class T, size_t N>
    constexpr std::array<T, N> TryStableSortKeys(std::array<T, N> v) {
        // Компилятор обычно ограничивает число шагов, которые можно сделать при вычислении constexpr-функции (см. опции `-fconstexpr-steps=N` или `/constexpr:stepsN`).
        // Число же шагов, необходимых для сортировки, зависит не только от длины массива,
        // но и от используемого алгоритма, от числа вложенных функций в его реализации, и от наличия assert'ов в ней.
        // Что также означает, что число шагов может меняться в несколько раз при смене реализации STL или при сборке с NDEBUG и без.
        //
        // Исчерпание бюджета на действия приведёт к ошибки компиляции без возможности восстановления.
        // То есть без возможности обнаружить эту ситуацию и переключится на другой алгоритм.
        //
        // Поэтому максимальный размер сортируемого массива заранее ограничивается безопасной константой.
        // А все массивы большего размера досортировываются уже во время исполнения программы.
        constexpr size_t MAX_COMPILE_TIME_SORT_ARRAY_SIZE = 2'000;
        if (v.size() > MAX_COMPILE_TIME_SORT_ARRAY_SIZE) {
            return v;
        }

        // Многие перечисления уже отсортированы. Но текущая реализация constexpr std::sort в libcxx не проверяет этот случай и всегда работает за время Θ(NlogN)
        if (IsSortedBy(v.begin(), v.end(), std::mem_fn(&T::Key))) {
            return v;
        }

        std::array<const T*, N> ptrs;
        Iota(ptrs.begin(), ptrs.end(), &v[0]);
        auto cmpKeyPointersFn = [](const T* a, const T* b) {
            if (a->Key != b->Key) {
                return a->Key < b->Key;
            }
            return a < b; // ensure stable sort order
        };
        Sort(ptrs.begin(), ptrs.end(), cmpKeyPointersFn);

        std::array<T, N> r;
        for (size_t i = 0; i < N; ++i) {
            r[i] = *ptrs[i];
        }
        return r;
    }

#else

    template <class T, size_t N>
    constexpr std::array<T, N> TryStableSortKeys(std::array<T, N> v) {
        return v; // skip sort in case of old language version
    }

#endif
}
