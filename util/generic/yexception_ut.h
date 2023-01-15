#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    typedef void (*TCallbackFun)(int);

    //! just calls callback with parameter @c i
    void TestCallback(TCallbackFun f, int i);

#ifdef __cplusplus
}
#endif
