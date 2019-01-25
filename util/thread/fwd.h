#pragma once

#include <stlfwd>

struct TDefaultLFCounter;

template <class T, class TCounter = TDefaultLFCounter>
class TLockFreeQueue;

template <class T, class TCounter = TDefaultLFCounter>
class TAutoLockFreeQueue;

template <class T>
class TLockFreeStackeue;

class IThreadFactory;

struct IObjectInQueue;
class TThreadFactoryHolder;

using TThreadFunction = std::function<void()>;

class IMtpQueue;
class TFakeMtpQueue;
class TMtpQueue;
class TAdaptiveMtpQueue;
class TSimpleMtpQueue;

template <class TQueueType, class TSlave>
class TMtpQueueBinder;
