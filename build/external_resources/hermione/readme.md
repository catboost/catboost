# hermione bundle

Ресурс должен быть tar.gz-архивом с пакетом `hermione`, `@yandex-int/hermione-cli` и поддерживаемыми hermione-плагинами. Каждый пакет поставляется со всеми транзитивными зависимостями. Структура:

```
node_modules/
    .bin/
        hermione
        hermione-cli
    @yandex-int/
        hermione-cli/
    hermione/
    html-reporter/
    hermione-chunks/
    hermione-passive-browsers/
```
