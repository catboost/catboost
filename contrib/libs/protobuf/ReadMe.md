документация к google protobuf: https://developers.google.com/protocol-buffers/

из ответа в @devtools об отличиях arcadia/contrib/libs/protobuf от google'овских:

@heretic:
> <...> в нашем протобуфе std::string заменен на TString <...>. Гугловая документация в целом применима.

@vvvv:
> <...> игнорирование корректности utf8 в string <...>

@orivej:
> чтобы закомитить в contrib/libs/protobuf нужное кому-либо изменение, не требуется знакомиться с прошлыми изменениями или обновлять список отличий, поэтому списка нет и никто все не знает. Кроме того, многие изменения не касаются использования. Однако рядом с некоторыми изменениями упоминается Yandex: https://cs.yandex-team.ru/#!Yandex,%5Econtrib%2Flibs%2Fprotobuf,,arcadia
> Из diff'а между аркадией и upstream'ом кроме уже названного видны:
> \- дополнительные api messagext.h и messagext_lite.h
>   https://a.yandex-team.ru/arc/trunk/arcadia/contrib/libs/protobuf/messagext.h?rev=3839146
>   https://a.yandex-team.ru/arc/trunk/arcadia/contrib/libs/protobuf/messagext_lite.h?rev=3839146
> \- дополнительное api json_util.h
>   https://a.yandex-team.ru/arc/commit/2442163
> \- новые аргументы --encode-decode-input и --encode-decode-output
>   https://a.yandex-team.ru/arc/commit/3839146
> \- удаление warning про отсутствие директивы syntax
>   https://a.yandex-team.ru/arc/commit/2442194
> \- добавление warning про дефис в пути python-модуля
>   https://a.yandex-team.ru/arc/commit/1491136


