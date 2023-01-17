# hermione bundle

Для обновления зависимостей ресурса необходимо использовать команду `create-external-resource` из `tools/nots` -  https://a.yandex-team.ru/svn/trunk/arcadia/tools/nots.
Для этого из текущей папки вызываем сборку nots командой - `ya make ../../../tools/nots` и удаляем `ya.make` (он перегенерится).
После чего вызываем создание ресурса с помощью команды: `../../../tools/nots/nots create-external-resource --external-resources-meta "meta.json" --yamake-owner "g:hermione" --resource-owner "HERMIONE"`.

В результате сгенерится ресурс с пакетами указанными в package.json.
