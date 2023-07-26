from gevent._gevent_c_greenlet_primitives cimport SwitchOutGreenletWithLoop

cdef _threadlocal

cpdef get_hub_class()
cpdef SwitchOutGreenletWithLoop get_hub_if_exists()
cpdef set_hub(SwitchOutGreenletWithLoop hub)
cpdef get_loop()
cpdef set_loop(loop)

cpdef SwitchOutGreenletWithLoop get_hub()

# XXX: TODO: Move the definition of TrackedRawGreenlet
# into a file that can be cython compiled so get_hub can
# return that.
cpdef SwitchOutGreenletWithLoop get_hub_noargs()
