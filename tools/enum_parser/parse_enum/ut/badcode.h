#pragma once

// Anonymous namespaces are meaningless, but should not break our parser
namespace {
    enum ETest {
        Http = 9 /* "http://" "secondary" "old\nvalue" */,
        Https = 1 /* "https://" */,
        ETestItemCount,
    };
}
