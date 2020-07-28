# Chromium trace viewer-compatible tracing

- [Overview](https://www.chromium.org/developers/how-tos/trace-event-profiling-tool)
- [Trace Viewer](https://github.com/catapult-project/catapult/tree/master/tracing)
- [Trace JSON Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview)

![Example](https://jing.yandex-team.ru/files/borman/browsertracing_2016-04-08_02-22-17.png)

```cpp
#include <library/cpp/chromium_trace/interface.h>

void TracedFunction() {
    CHROMIUM_TRACE_FUNCTION();

    ...
}

int main() {
    NChromiumTrace::TGlobalJsonFileSink traceSink("trace.json");

    ...
}
```
