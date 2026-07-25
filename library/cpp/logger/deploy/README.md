# Deploy JSON backend для `library/cpp/logger`

Пишет логи в [формате Yandex Deploy](https://deploy.yandex-team.ru/docs/logs/format)
(JSON-строки для Unified Agent / Monitoring).

Аналоги: Python `library/python/deploy_formatter`, Go `library/go/core/log/zap`
(`NewDeployLogger`), TypeScript `yandex-logger` (deploy stream).

## Использование

```cpp
#include <library/cpp/logger/deploy/backend.h>
#include <library/cpp/logger/log.h>
#include <library/cpp/logger/stream.h>

TLog log(MakeHolder<TDeployJsonLogBackend>(
    MakeHolder<TStreamLogBackend>(&Cout),
    "my-service"));

TLogElement(&log, TLOG_INFO)
    .With("request_id", "abc")
    .With("code", "200")
    << "hello";
```

```json
{"@timestamp":"...","levelStr":"INFO","message":"hello","loggerName":"my-service","request_id":"abc","@fields":{"code":"200"}}
```

`loggerName` задаётся в конструкторе. Через `With(...)` в корень JSON попадают
`request_id`, `user_id`, `stackTrace`, `threadName`; остальное — в `@fields`.
