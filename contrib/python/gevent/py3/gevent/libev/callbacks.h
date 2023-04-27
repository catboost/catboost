struct ev_loop;
struct PyGeventLoopObject;
struct PyGeventCallbackObject;

#define DEFINE_CALLBACK(WATCHER_LC, WATCHER_TYPE) \
    void gevent_callback_##WATCHER_LC(struct ev_loop *, void *, int);


#define DEFINE_CALLBACKS0              \
    DEFINE_CALLBACK(io, IO);           \
    DEFINE_CALLBACK(timer, Timer);     \
    DEFINE_CALLBACK(signal, Signal);   \
    DEFINE_CALLBACK(idle, Idle);       \
    DEFINE_CALLBACK(prepare, Prepare); \
    DEFINE_CALLBACK(check, Check);     \
    DEFINE_CALLBACK(fork, Fork);       \
    DEFINE_CALLBACK(async, Async);     \
    DEFINE_CALLBACK(stat, Stat);       \
    DEFINE_CALLBACK(child, Child);


#define DEFINE_CALLBACKS DEFINE_CALLBACKS0


DEFINE_CALLBACKS


void gevent_run_callbacks(struct ev_loop *, void *, int);

void gevent_call(struct PyGeventLoopObject* loop, struct PyGeventCallbackObject* cb);

void* gevent_realloc(void* ptr, size_t size);
void gevent_noop(struct ev_loop*, void* watcher, int revents);

/* Only used on Win32 */
void gevent_periodic_signal_check(struct ev_loop *, void *, int);

// We're included in corecext.c. Disable a bunch of annoying warnings
// that are in the generated code that we can't do anything about.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunreachable-code"
#endif
