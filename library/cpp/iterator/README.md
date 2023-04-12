# Functools library

Эта библиотека предоставляет функции `Enumerate`, `Zip`, `Map`, `Filter`, `Concatenate` и `CartesianProduct`.

`Enumerate`, `Zip`, `Map`, `Filter` повторяют логику одноименных функций из python.

Важный момент:
 * Итераторы данных view почти во всех случаях (кроме Map, там зависит от маппера) возвращают `std::tuple` **по значению** (при этом шаблонные параметры всегда, когда это уместно, ссылки). 
   <br> Так что нет никакого смысла делать `for (const auto& [i, x] : Enumerate(c))`. 
   <br> Лучше всегда `for (auto [i, x] : Enumerate(c))` (`x` в этом случае будет ссылкой и работать с ним можно как со ссылкой) или `for (const auto [i, x] : Enumerate(c))`.

Предоставляемые гарантии:
 * Работа для всех контейнеров, для которых работает range-based for (для `Enumerate`, `Zip`, `Concatenate`, `CartesianProduct`).
   Для `Map` и `Filter` есть требование на то, что первый и последний итераторы контейнера имеют один тип
 * В том числе работает для обычных массивов (`int a[] = {1, 2, 3}; Enumerate(a)`).
 * Поддержка rvalue для контейнеров, предикатов и функций-мапперов (`Filter([](auto x){...}, TVector{1, 2, 3})`).
   В этом случае объекты сохраняются внутри view.
 * Проброс элементов контейнеров по неконстантной ссылке
 * `TView::iterator` - можно полагаться, что этот тип есть и корректен
 * `TIterator::iterator_category` - можно полагаться, что этот тип есть и определен.


На что гарантий нет:
 * Любые изменения контейнеров, меняющие размер или инвалидирующие их итераторы, инвалидируют созданные view
 * Не принимает списки инициализации (`Enumerate({1, 2, 3})`), так как неизвестен желаемый тип контейнера.
 * В классах реализации оставлены публичные члены вида `.Field_`, чтобы не загромождать реализацию
   Тем не менее эти поля не гарантированы, могут стать приватными или исчезнуть
 * Для всех итераторов определены вложенные типы: `value_type`, `pointer`, `reference`.
   Тем не менее не рекомендуется их использовать в связи с их неоднозначностью.
   `value_type` может быть как обычным типом, так и ссылкой. Может быть `std::tuple<T1, T2>`,
   а может `std::tuple<T1&, const T2&>`.
   Если возникает необходимость в этих типах, то возможно, стоит упростить код и вообще не использовать эти view.
   Если очень хочется можно использовать `delctype(*container.begin())`.


Производительность:
 * Бенчмарки времени компиляции и скорости выполнения, а так же сравнение с range-v3 и другими существующими реализациями
   доступны в [репозитории где ведется разработка](https://github.com/yuri-pechatnov/cpp_functools/tree/master "functools").



Q: Оверхед?
A: По выполнению: на Enumerate, Zip, Map - нулевой. Где-то x1.5 на Filter, и x3 на Concatenate и CartesianProduct. Но если в теле цикла происходит хоть что-то существенное, то это пренебрежимо мало.
   По компиляции: сложно рассчитать как оно скажется в реальном большом проекте, но приблизительно не более x1.5 на один цикл.

Q: А зачем свой велосипед?
A: ((https://pechatnov.at.yandex-team.ru/67 Ответ в этом посте)).

Q: А почему вот здесь плохо написано, надо же по-другому?
A: Код несколько раз переписывался и согласовывался ((https://st.yandex-team.ru/IGNIETFERRO-973 более полугода)). А допиливать его внутреннюю реализацию после коммита никто не мешает и дальше.


Сигнатуры и эквиваленты:


```cpp
//! In all comments variables ending with '_'
//! are considered as invisible for {...} block.

//! Usage: for (auto [i, x] : Enumerate(container)) {...}
//! Equivalent: { std::size_t i_ = 0; for (auto& x : container) { const std::size_t i = i_; {...}; ++i_; }}
template <typename TContainerOrRef>
auto Enumerate(TContainerOrRef&& container);

//! Usage: for (auto x : Map(mapperFunc, container)) {...}
//! Equivalent: for (auto iter_ = std::begin(container); iter_ != std::end(container); ++iter_) {
//!                 auto x = mapperFunc(*iter_); {...}; }
template <typename TMapper, typename TContainerOrRef>
auto Map(TMapper&& mapper, TContainerOrRef&& container);

//! Usage: for (auto x : Filter(predicate, container)) {...}
//! Equivalent: for (auto x : container) { if (predicate(x)) {...}}
template <typename TPredicate, typename TContainerOrRef>
auto Filter(TPredicate&& predicate, TContainerOrRef&& container);

//! Usage: for (auto [ai, bi] : Zip(a, b)) {...}
//! Equivalent: { auto ia_ = std::begin(a); auto ib_ = std::begin(b);
//!               for (; ia_ != std::end(a) && ib_ != std::end(b); ++ia_, ++ib_) {
//!                   auto&& ai = *ia_; auto&& bi = *ib_; {...}
//!               }}
template <typename... TContainers>
auto Zip(TContainers&&... containers);

//! Usage: for (auto x : Reversed(container)) {...}
//! Equivalent: for (auto iter_ = std::rbegin(container); iter_ != std::rend(container); ++iter_) {
//!                 auto x = *iter_; {...}}
template <typename TContainerOrRef>
auto Reversed(TContainerOrRef&& container);

//! Usage: for (auto x : Concatenate(a, b)) {...}
//! Equivalent: { for (auto x : a) {...} for (auto x : b) {...} }
//! (if there is no static variables in {...})
template <typename TFirstContainer, typename... TContainers>
auto Concatenate(TFirstContainer&& container, TContainers&&... containers);

//! Usage: for (auto [ai, bi] : CartesianProduct(a, b)) {...}
//! Equivalent: for (auto& ai : a) { for (auto& bi : b) {...} }
template <typename... TContainers>
auto CartesianProduct(TContainers&&... containers);
```
