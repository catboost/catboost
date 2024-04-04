/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Written and engineered 2008-2021 at the University of Edinburgh    */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/*    Authors: Julian Hall, Ivet Galabova, Qi Huangfu, Leona Gottwald    */
/*    and Michael Feldmeier                                              */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HIGHS_SPIN_MUTEX_H_
#define HIGHS_SPIN_MUTEX_H_

#include <atomic>

#include "HConfig.h"

#ifdef HIGHS_HAVE_MM_PAUSE
#include <immintrin.h>
#else
#include <thread>
#endif

class HighsSpinMutex {
  std::atomic<bool> flag{false};

 public:
  static void yieldProcessor() {
#ifdef HIGHS_HAVE_MM_PAUSE
    _mm_pause();
#else
    std::this_thread::yield();
#endif
  }

  bool try_lock() { return !flag.exchange(true, std::memory_order_acquire); }

  void lock() {
    while (true) {
      if (!flag.exchange(true, std::memory_order_acquire)) return;

      while (flag.load(std::memory_order_relaxed)) yieldProcessor();
    }
  }

  void unlock() { flag.store(false, std::memory_order_release); }
};

#endif