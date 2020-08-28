#pragma once

#define weak __attribute__((__weak__))
#define weak_alias(old, new) \
        extern __typeof(old) new __attribute__((__weak__, __alias__(#old)))
