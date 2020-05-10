Бинарная куча с доступом по ключу и произвольным изменением приоритета этих ключей
=====================================================
Плюсовый аналог питоновского https://pypi.python.org/pypi/HeapDict

Одновременно хэш-мап (THashMap) и куча (priority_queue)
Методы кучи:
- top() - возвращает пару (ключ, приоритет) с наибольшим приоритетом
- pop() - удаляет из кучи top()
- push(key, priority)

Методы хэш-мапа:
- operator[](key)
- find(key)
- erase(key), erase(iterator)
- insert(std::make_pair(key, priority))

Все методы работают за O(log(N)). Но скрытая константа очень маленькая (см. benchmark).

Позволяет делать, например, вот так:
```
TVector<TString> GetNMostFrequentWords(TInputStream &in, size_t N) {
    THeapDict<TString, size_t> heapDict;

    while (TString line = in.ReadLine()) {
        TString word;
        size_t count;
        Split(line, ' ', word, count);
        heapDict[word] += count;
    }

    TVector<TString> result;
    while (result.size() < N && !heapDict.empty()) {
        result.push_back(heapDict.top().first);
        heapDict.pop();
    }
    return result;
}
```

У данной структуры данных есть одна важная особенность - у нее нет const-методов для доступа к данным.
То есть, имея константную ссылку на THeapDict, можно будет узнать только его .size().

Эта особенность - необходимая жертва ради удобного интерфейса, имитирующего map.
Более конкретно, если пользователь меняет приоритет ключа
```
map[key] = value;
```
то структура данных среагирует на это изменение (перестроит кучу) лениво, т.е. при следующем доступе к элементам.

Из-за этой же особенности проитерироваться по структуре данных можно только так:
```
for (THeapDict<TString, int>::const_iterator it = heapDict.cbegin(); it != heapDict.cend(); ++it) {
    // ...
}
```
То есть, только с ```const_iterator``` и только с ```cbegin()/cend()```.
