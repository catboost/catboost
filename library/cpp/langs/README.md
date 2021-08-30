Здесь описаны константы для [языков](https://a.yandex-team.ru/arc/trunk/arcadia/library/cpp/langs/langs.h) и [письменностей](https://a.yandex-team.ru/arc/trunk/arcadia/library/cpp/langs/scripts.h) (скриптов в терминах Unicode).

В терминах этих констант языков работают [документная](https://a.yandex-team.ru/arc/trunk/arcadia/kernel/recshell/recshell.h) и [запросная](https://a.yandex-team.ru/arc/trunk/arcadia/dict/recognize/queryrec) распознавалки языка.

Имеется [набор функций](https://a.yandex-team.ru/arc/trunk/arcadia/library/cpp/langs/langs.h?rev=r6909333#L142-214) для преобразования констант в двухбуквенный или трехбуквенный код и обратного получения константы по строке с учетом синонимов. Есть [функции](https://a.yandex-team.ru/arc/trunk/arcadia/library/cpp/langs/langs.h?rev=r6909333#L216-217) для определения письменности по языку и по символу).

В списке констант представлены не все языки и письменности, а лишь те, которые представляли интерес для поиска Яндекса и машинного перевода.
Имеется несколько псевдоязыков типа `LANG_UZB_CYR` или `LANG_KAZ_LAT`.
