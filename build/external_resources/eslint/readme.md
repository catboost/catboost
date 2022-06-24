# eslint bundle

Ресурс должен быть tar.gz-архивом с пакетами `eslint` и `@yandex-int/lint` и всеми транзитивными зависимостями. Структура:

```
node_modules/
    .bin/
        eslint
    @yandex-int/
        lint/
    eslint/
    остальные пакеты, нужные для eslint
```
