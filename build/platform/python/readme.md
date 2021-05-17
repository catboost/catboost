# Инструкция по добавлению нового бандла с системным питоном

## Майним бандлы системного питона
Бандлы системного питон майнятся для трех платформ: linux, darwin, windows.
Подставляем под PYTHON_VERSION - версию нужного питона
### Linux

1. Устанавливаем систему с версией ubuntu, из которой планируется брать системный питон. Здесь есть два варианта
    1. Если нужно собрать системный питон, который будет запускать тесты на дистбилде, то нужно использовать ubuntu такой же версии, что и на дистбилде.
    Тут стоит учитывать, что на дистбилде может быть достаточно старая версия ubuntu, на котором не будет нужной версии питона.
    2. Выбрать ту версию ubuntu, в которой есть нужный питон
2. `mkdir -p ~/work/packages`
3. `cd ~/work/packages`
4. майним deb-пакеты питона
    1. Майним системный питон для запуска на дистбилде:

            apt-get download $(apt-cache depends --recurse --no-recommends --no-suggests --no-conflicts --no-breaks --no-replaces --no-enhances python{PYTHON_VERSION}-dev python{|3}-pkg-resources python{|3}-setuptools | grep "^\w" | sort -u)
            rm libc6* libc-*

    2. Майним системный питон для сборки сошек:

            apt download python{PYTHON_VERSION} python{PYTHON_VERSION}-dev python{PYTHON_VERSION}-minimal libpython{PYTHON_VERSION} libpython{PYTHON_VERSION}-dev libpython{PYTHON_VERSION}-stdlib libpython{PYTHON_VERSION}-minimal

5. `cd ..`
6. `for path in $(ls packages); do ar -xf packages/$path; tar -xf data.tar.xz; done;`
7. `mv usr python`
8. `tar -czf python{PYTHON_VERSION}_linux.tar.gz python`
9. `ya upload python{PYTHON_VERSION}_linux.tar.gz -d "Ubuntu {UBUNTU_VERSION} x86_64 python{PYTHON_VERSION} installation" --do-not-remove`

UBUNTU_VERSION - версия ubuntu, на которой майнился системный питон

### Darwin
1. Находим macbook.
2. Все установленные питоны лежат в `/Library/Frameworks/Python.framework/Versions`
3. Копируем `/Library/Frameworks/Python.framework` в директорию с именем `python`
4. Чистим `python/Python.framework/Versions/` от ненужных питонов
5. Проверяем, что симлинки указывают в правильные места
    1. `python/Python.framework/Versions/Current -> {PYTHON_VERSION}`
    2. `python/Python.framework/Headers -> Versions/Current/Headers`
    3. `python/Python.framework/Python -> Versions/Current/Python`
    4. `python/Python.framework/Resources -> Versions/Current/Resources`
6. `tar -czf python{PYTHON_VERSION}_darwin.tar.gz python`
7. `ya upload python{PYTHON_VERSION}_darwin.tar.gz -d "Darwin x86_64 python{PYTHON_VERSION} installation" --do-not-remove`

Если нужного питона нет в системе, его нужно установить из `python.org`, его установку можно найти в стандартном месте.

Если нужен питон из `brew`, его установку можно найти в `/usr/local/Cellar/python*/{python_version}/Frameworks/`,
 а дальше следовать стандартной инструкции

### Windows
1. Находим машинку с windows
2. Устанавливаем нужную версию питона из `python.org`
3. Копируем содержимое установки питона в директорию `python`
4. Пакуем директорию `python` в `python{PYTHON_VERSION}_windows.tar.gz`
5. `ya upload python{PYTHON_VERSION}_windows.tar.gz -d "Windows x86_64 python{PYTHON_VERSION} installation" --do-not-remove`

## Добавляем бандлы системного питона в сборку

1. Конфигурация бандлов системных питонов находится здесь [build/platform/python](https://a.yandex-team.ru/arc/trunk/arcadia/build/platform/python)
2. Добавляем сендбокс ресурсы собранных бандлов в файл [resources.inc](https://a.yandex-team.ru/arc/trunk/arcadia/build/platform/python/resources.inc)

        SET(PYTHON38_LINUX sbr:1211259884)

3. Добавляем служебные переменные `_SYSTEM_PYTHON*, PY_VERSION, PY_FRAMEWORK_VERSION` для системного питона, если их еще нет,
в [ymake.core.conf](https://a.yandex-team.ru/arc/trunk/arcadia/build/ymake.core.conf?rev=7640792#L380) по аналогии.

        "3.8" ? {
            _SYSTEM_PYTHON38=yes
            PY_VERSION=3.8
            PY_FRAMEWORK_VERSION=3.8
        }

4. Добавляем ресурс в [build/platform/python/ya.make](https://a.yandex-team.ru/arc/trunk/arcadia/build/platform/python/ya.make)

          DECLARE_EXTERNAL_RESOURCE(EXTERNAL_PYTHON ${PYTHON38_LINUX})

## Проверяем сборку
1. Создаем тривиальный PY2MODULE с использование `c api` положенного питона, или находим подходящий в репозитории
2. Собираем его:
    1. linux `ya make -DUSE_SYSTEM_PYTHON=3.8 --target-platform linux`
    2. darwin `ya make -DUSE_SYSTEM_PYTHON=3.8 --target-platform darwin`
    3. windows `ya make -DUSE_SYSTEM_PYTHON=3.8 --target-platform win`
3. Проверяем, что получившиеся модули импортятся в питонах на соответсвующих системах
