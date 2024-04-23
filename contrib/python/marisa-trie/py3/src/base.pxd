cdef extern from "<marisa/base.h>":

    # A dictionary consists of 3 tries in default. Usually more tries make a
    # dictionary space-efficient but time-inefficient.
    ctypedef enum marisa_num_tries:
        MARISA_MIN_NUM_TRIES
        MARISA_MAX_NUM_TRIES
        MARISA_DEFAULT_NUM_TRIES


    # This library uses a cache technique to accelerate search functions. The
    # following enumerated type marisa_cache_level gives a list of available cache
    # size options. A larger cache enables faster search but takes a more space.
    ctypedef enum marisa_cache_level:
        MARISA_HUGE_CACHE
        MARISA_LARGE_CACHE
        MARISA_NORMAL_CACHE
        MARISA_SMALL_CACHE
        MARISA_TINY_CACHE
        MARISA_DEFAULT_CACHE

    # This library provides 2 kinds of TAIL implementations.
    ctypedef enum marisa_tail_mode:
        # MARISA_TEXT_TAIL merges last labels as zero-terminated strings. So, it is
        # available if and only if the last labels do not contain a NULL character.
        # If MARISA_TEXT_TAIL is specified and a NULL character exists in the last
        # labels, the setting is automatically switched to MARISA_BINARY_TAIL.
        MARISA_TEXT_TAIL

        # MARISA_BINARY_TAIL also merges last labels but as byte sequences. It uses
        # a bit vector to detect the end of a sequence, instead of NULL characters.
        # So, MARISA_BINARY_TAIL requires a larger space if the average length of
        # labels is greater than 8.
        MARISA_BINARY_TAIL

        MARISA_DEFAULT_TAIL

    # The arrangement of nodes affects the time cost of matching and the order of
    # predictive search.
    ctypedef enum marisa_node_order:
        # MARISA_LABEL_ORDER arranges nodes in ascending label order.
        # MARISA_LABEL_ORDER is useful if an application needs to predict keys in
        # label order.
        MARISA_LABEL_ORDER

        # MARISA_WEIGHT_ORDER arranges nodes in descending weight order.
        # MARISA_WEIGHT_ORDER is generally a better choice because it enables faster
        # matching.
        MARISA_WEIGHT_ORDER
        MARISA_DEFAULT_ORDER

    ctypedef enum marisa_config_mask:
        MARISA_NUM_TRIES_MASK
        MARISA_CACHE_LEVEL_MASK
        MARISA_TAIL_MODE_MASK
        MARISA_NODE_ORDER_MASK
        MARISA_CONFIG_MASK


cdef extern from "<marisa/base.h>" namespace "marisa":
    ctypedef marisa_cache_level CacheLevel
    ctypedef marisa_tail_mode TailMode
    ctypedef marisa_node_order NodeOrder
