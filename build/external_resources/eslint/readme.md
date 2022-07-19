# eslint bundle

Ресурс должен быть tar.gz-архивом с пакетами `eslint` и `@yandex-int/eslint-plugin-config` и всеми транзитивными зависимостями. Структура:

```
node_modules/
    .bin/
        eslint
    @yandex-int/
        eslint-plugin-config/
    eslint/
    остальные пакеты, нужные для eslint
```
