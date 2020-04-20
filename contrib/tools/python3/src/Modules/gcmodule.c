/*

  Reference Cycle Garbage Collection
  ==================================

  Neil Schemenauer <nas@arctrix.com>

  Based on a post on the python-dev list.  Ideas from Guido van Rossum,
  Eric Tiedemann, and various others.

  http://www.arctrix.com/nas/python/gc/

  The following mailing list threads provide a historical perspective on
  the design of this module.  Note that a fair amount of refinement has
  occurred since those discussions.

  http://mail.python.org/pipermail/python-dev/2000-March/002385.html
  http://mail.python.org/pipermail/python-dev/2000-March/002434.html
  http://mail.python.org/pipermail/python-dev/2000-March/002497.html

  For a highlevel view of the collection process, read the collect
  function.

*/

#include "Python.h"
#include "internal/context.h"
#include "internal/mem.h"
#include "internal/pystate.h"
#include "frameobject.h"        /* for PyFrame_ClearFreeList */
#include "pydtrace.h"
#include "pytime.h"             /* for _PyTime_GetMonotonicClock() */

/*[clinic input]
module gc
[clinic start generated code]*/
/*[clinic end generated code: output=da39a3ee5e6b4b0d input=b5c9690ecc842d79]*/

/* Get an object's GC head */
#define AS_GC(o) ((PyGC_Head *)(o)-1)

/* Get the object given the GC head */
#define FROM_GC(g) ((PyObject *)(((PyGC_Head *)g)+1))

/* Python string to use if unhandled exception occurs */
static PyObject *gc_str = NULL;

/* set for debugging information */
#define DEBUG_STATS             (1<<0) /* print collection statistics */
#define DEBUG_COLLECTABLE       (1<<1) /* print collectable objects */
#define DEBUG_UNCOLLECTABLE     (1<<2) /* print uncollectable objects */
#define DEBUG_SAVEALL           (1<<5) /* save all garbage in gc.garbage */
#define DEBUG_LEAK              DEBUG_COLLECTABLE | \
                DEBUG_UNCOLLECTABLE | \
                DEBUG_SAVEALL

#define GEN_HEAD(n) (&_PyRuntime.gc.generations[n].head)

void
_PyGC_Initialize(struct _gc_runtime_state *state)
{
    state->enabled = 1; /* automatic collection enabled? */

#define _GEN_HEAD(n) (&state->generations[n].head)
    struct gc_generation generations[NUM_GENERATIONS] = {
        /* PyGC_Head,                                 threshold,      count */
        {{{_GEN_HEAD(0), _GEN_HEAD(0), 0}},           700,            0},
        {{{_GEN_HEAD(1), _GEN_HEAD(1), 0}},           10,             0},
        {{{_GEN_HEAD(2), _GEN_HEAD(2), 0}},           10,             0},
    };
    for (int i = 0; i < NUM_GENERATIONS; i++) {
        state->generations[i] = generations[i];
    };
    state->generation0 = GEN_HEAD(0);
    struct gc_generation permanent_generation = {
          {{&state->permanent_generation.head, &state->permanent_generation.head, 0}}, 0, 0
    };
    state->permanent_generation = permanent_generation;
}

/*--------------------------------------------------------------------------
gc_refs values.

Between collections, every gc'ed object has one of two gc_refs values:

GC_UNTRACKED
    The initial state; objects returned by PyObject_GC_Malloc are in this
    state.  The object doesn't live in any generation list, and its
    tp_traverse slot must not be called.

GC_REACHABLE
    The object lives in some generation list, and its tp_traverse is safe to
    call.  An object transitions to GC_REACHABLE when PyObject_GC_Track
    is called.

During a collection, gc_refs can temporarily take on other states:

>= 0
    At the start of a collection, update_refs() copies the true refcount
    to gc_refs, for each object in the generation being collected.
    subtract_refs() then adjusts gc_refs so that it equals the number of
    times an object is referenced directly from outside the generation
    being collected.
    gc_refs remains >= 0 throughout these steps.

GC_TENTATIVELY_UNREACHABLE
    move_unreachable() then moves objects not reachable (whether directly or
    indirectly) from outside the generation into an "unreachable" set.
    Objects that are found to be reachable have gc_refs set to GC_REACHABLE
    again.  Objects that are found to be unreachable have gc_refs set to
    GC_TENTATIVELY_UNREACHABLE.  It's "tentatively" because the pass doing
    this can't be sure until it ends, and GC_TENTATIVELY_UNREACHABLE may
    transition back to GC_REACHABLE.

    Only objects with GC_TENTATIVELY_UNREACHABLE still set are candidates
    for collection.  If it's decided not to collect such an object (e.g.,
    it has a __del__ method), its gc_refs is restored to GC_REACHABLE again.
----------------------------------------------------------------------------
*/
#define GC_UNTRACKED                    _PyGC_REFS_UNTRACKED
#define GC_REACHABLE                    _PyGC_REFS_REACHABLE
#define GC_TENTATIVELY_UNREACHABLE      _PyGC_REFS_TENTATIVELY_UNREACHABLE

#define IS_TRACKED(o) (_PyGC_REFS(o) != GC_UNTRACKED)
#define IS_REACHABLE(o) (_PyGC_REFS(o) == GC_REACHABLE)
#define IS_TENTATIVELY_UNREACHABLE(o) ( \
    _PyGC_REFS(o) == GC_TENTATIVELY_UNREACHABLE)

/*** list functions ***/

static void
gc_list_init(PyGC_Head *list)
{
    list->gc.gc_prev = list;
    list->gc.gc_next = list;
}

static int
gc_list_is_empty(PyGC_Head *list)
{
    return (list->gc.gc_next == list);
}

#if 0
/* This became unused after gc_list_move() was introduced. */
/* Append `node` to `list`. */
static void
gc_list_append(PyGC_Head *node, PyGC_Head *list)
{
    node->gc.gc_next = list;
    node->gc.gc_prev = list->gc.gc_prev;
    node->gc.gc_prev->gc.gc_next = node;
    list->gc.gc_prev = node;
}
#endif

/* Remove `node` from the gc list it's currently in. */
static void
gc_list_remove(PyGC_Head *node)
{
    node->gc.gc_prev->gc.gc_next = node->gc.gc_next;
    node->gc.gc_next->gc.gc_prev = node->gc.gc_prev;
    node->gc.gc_next = NULL; /* object is not currently tracked */
}

/* Move `node` from the gc list it's currently in (which is not explicitly
 * named here) to the end of `list`.  This is semantically the same as
 * gc_list_remove(node) followed by gc_list_append(node, list).
 */
static void
gc_list_move(PyGC_Head *node, PyGC_Head *list)
{
    PyGC_Head *new_prev;
    PyGC_Head *current_prev = node->gc.gc_prev;
    PyGC_Head *current_next = node->gc.gc_next;
    /* Unlink from current list. */
    current_prev->gc.gc_next = current_next;
    current_next->gc.gc_prev = current_prev;
    /* Relink at end of new list. */
    new_prev = node->gc.gc_prev = list->gc.gc_prev;
    new_prev->gc.gc_next = list->gc.gc_prev = node;
    node->gc.gc_next = list;
}

/* append list `from` onto list `to`; `from` becomes an empty list */
static void
gc_list_merge(PyGC_Head *from, PyGC_Head *to)
{
    PyGC_Head *tail;
    assert(from != to);
    if (!gc_list_is_empty(from)) {
        tail = to->gc.gc_prev;
        tail->gc.gc_next = from->gc.gc_next;
        tail->gc.gc_next->gc.gc_prev = tail;
        to->gc.gc_prev = from->gc.gc_prev;
        to->gc.gc_prev->gc.gc_next = to;
    }
    gc_list_init(from);
}

static Py_ssize_t
gc_list_size(PyGC_Head *list)
{
    PyGC_Head *gc;
    Py_ssize_t n = 0;
    for (gc = list->gc.gc_next; gc != list; gc = gc->gc.gc_next) {
        n++;
    }
    return n;
}

/* Append objects in a GC list to a Python list.
 * Return 0 if all OK, < 0 if error (out of memory for list).
 */
static int
append_objects(PyObject *py_list, PyGC_Head *gc_list)
{
    PyGC_Head *gc;
    for (gc = gc_list->gc.gc_next; gc != gc_list; gc = gc->gc.gc_next) {
        PyObject *op = FROM_GC(gc);
        if (op != py_list) {
            if (PyList_Append(py_list, op)) {
                return -1; /* exception */
            }
        }
    }
    return 0;
}

/*** end of list stuff ***/


/* Set all gc_refs = ob_refcnt.  After this, gc_refs is > 0 for all objects
 * in containers, and is GC_REACHABLE for all tracked gc objects not in
 * containers.
 */
static void
update_refs(PyGC_Head *containers)
{
    PyGC_Head *gc = containers->gc.gc_next;
    for (; gc != containers; gc = gc->gc.gc_next) {
        assert(_PyGCHead_REFS(gc) == GC_REACHABLE);
        _PyGCHead_SET_REFS(gc, Py_REFCNT(FROM_GC(gc)));
        /* Python's cyclic gc should never see an incoming refcount
         * of 0:  if something decref'ed to 0, it should have been
         * deallocated immediately at that time.
         * Possible cause (if the assert triggers):  a tp_dealloc
         * routine left a gc-aware object tracked during its teardown
         * phase, and did something-- or allowed something to happen --
         * that called back into Python.  gc can trigger then, and may
         * see the still-tracked dying object.  Before this assert
         * was added, such mistakes went on to allow gc to try to
         * delete the object again.  In a debug build, that caused
         * a mysterious segfault, when _Py_ForgetReference tried
         * to remove the object from the doubly-linked list of all
         * objects a second time.  In a release build, an actual
         * double deallocation occurred, which leads to corruption
         * of the allocator's internal bookkeeping pointers.  That's
         * so serious that maybe this should be a release-build
         * check instead of an assert?
         */
        assert(_PyGCHead_REFS(gc) != 0);
    }
}

/* A traversal callback for subtract_refs. */
static int
visit_decref(PyObject *op, void *data)
{
    assert(op != NULL);
    if (PyObject_IS_GC(op)) {
        PyGC_Head *gc = AS_GC(op);
        /* We're only interested in gc_refs for objects in the
         * generation being collected, which can be recognized
         * because only they have positive gc_refs.
         */
        assert(_PyGCHead_REFS(gc) != 0); /* else refcount was too small */
        if (_PyGCHead_REFS(gc) > 0)
            _PyGCHead_DECREF(gc);
    }
    return 0;
}

/* Subtract internal references from gc_refs.  After this, gc_refs is >= 0
 * for all objects in containers, and is GC_REACHABLE for all tracked gc
 * objects not in containers.  The ones with gc_refs > 0 are directly
 * reachable from outside containers, and so can't be collected.
 */
static void
subtract_refs(PyGC_Head *containers)
{
    traverseproc traverse;
    PyGC_Head *gc = containers->gc.gc_next;
    for (; gc != containers; gc=gc->gc.gc_next) {
        traverse = Py_TYPE(FROM_GC(gc))->tp_traverse;
        (void) traverse(FROM_GC(gc),
                       (visitproc)visit_decref,
                       NULL);
    }
}

/* A traversal callback for move_unreachable. */
static int
visit_reachable(PyObject *op, PyGC_Head *reachable)
{
    if (PyObject_IS_GC(op)) {
        PyGC_Head *gc = AS_GC(op);
        const Py_ssize_t gc_refs = _PyGCHead_REFS(gc);

        if (gc_refs == 0) {
            /* This is in move_unreachable's 'young' list, but
             * the traversal hasn't yet gotten to it.  All
             * we need to do is tell move_unreachable that it's
             * reachable.
             */
            _PyGCHead_SET_REFS(gc, 1);
        }
        else if (gc_refs == GC_TENTATIVELY_UNREACHABLE) {
            /* This had gc_refs = 0 when move_unreachable got
             * to it, but turns out it's reachable after all.
             * Move it back to move_unreachable's 'young' list,
             * and move_unreachable will eventually get to it
             * again.
             */
            gc_list_move(gc, reachable);
            _PyGCHead_SET_REFS(gc, 1);
        }
        /* Else there's nothing to do.
         * If gc_refs > 0, it must be in move_unreachable's 'young'
         * list, and move_unreachable will eventually get to it.
         * If gc_refs == GC_REACHABLE, it's either in some other
         * generation so we don't care about it, or move_unreachable
         * already dealt with it.
         * If gc_refs == GC_UNTRACKED, it must be ignored.
         */
         else {
            assert(gc_refs > 0
                   || gc_refs == GC_REACHABLE
                   || gc_refs == GC_UNTRACKED);
         }
    }
    return 0;
}

/* Move the unreachable objects from young to unreachable.  After this,
 * all objects in young have gc_refs = GC_REACHABLE, and all objects in
 * unreachable have gc_refs = GC_TENTATIVELY_UNREACHABLE.  All tracked
 * gc objects not in young or unreachable still have gc_refs = GC_REACHABLE.
 * All objects in young after this are directly or indirectly reachable
 * from outside the original young; and all objects in unreachable are
 * not.
 */
static void
move_unreachable(PyGC_Head *young, PyGC_Head *unreachable)
{
    PyGC_Head *gc = young->gc.gc_next;

    /* Invariants:  all objects "to the left" of us in young have gc_refs
     * = GC_REACHABLE, and are indeed reachable (directly or indirectly)
     * from outside the young list as it was at entry.  All other objects
     * from the original young "to the left" of us are in unreachable now,
     * and have gc_refs = GC_TENTATIVELY_UNREACHABLE.  All objects to the
     * left of us in 'young' now have been scanned, and no objects here
     * or to the right have been scanned yet.
     */

    while (gc != young) {
        PyGC_Head *next;

        if (_PyGCHead_REFS(gc)) {
            /* gc is definitely reachable from outside the
             * original 'young'.  Mark it as such, and traverse
             * its pointers to find any other objects that may
             * be directly reachable from it.  Note that the
             * call to tp_traverse may append objects to young,
             * so we have to wait until it returns to determine
             * the next object to visit.
             */
            PyObject *op = FROM_GC(gc);
            traverseproc traverse = Py_TYPE(op)->tp_traverse;
            assert(_PyGCHead_REFS(gc) > 0);
            _PyGCHead_SET_REFS(gc, GC_REACHABLE);
            (void) traverse(op,
                            (visitproc)visit_reachable,
                            (void *)young);
            next = gc->gc.gc_next;
            if (PyTuple_CheckExact(op)) {
                _PyTuple_MaybeUntrack(op);
            }
        }
        else {
            /* This *may* be unreachable.  To make progress,
             * assume it is.  gc isn't directly reachable from
             * any object we've already traversed, but may be
             * reachable from an object we haven't gotten to yet.
             * visit_reachable will eventually move gc back into
             * young if that's so, and we'll see it again.
             */
            next = gc->gc.gc_next;
            gc_list_move(gc, unreachable);
            _PyGCHead_SET_REFS(gc, GC_TENTATIVELY_UNREACHABLE);
        }
        gc = next;
    }
}

/* Try to untrack all currently tracked dictionaries */
static void
untrack_dicts(PyGC_Head *head)
{
    PyGC_Head *next, *gc = head->gc.gc_next;
    while (gc != head) {
        PyObject *op = FROM_GC(gc);
        next = gc->gc.gc_next;
        if (PyDict_CheckExact(op))
            _PyDict_MaybeUntrack(op);
        gc = next;
    }
}

/* Return true if object has a pre-PEP 442 finalization method. */
static int
has_legacy_finalizer(PyObject *op)
{
    return op->ob_type->tp_del != NULL;
}

/* Move the objects in unreachable with tp_del slots into `finalizers`.
 * Objects moved into `finalizers` have gc_refs set to GC_REACHABLE; the
 * objects remaining in unreachable are left at GC_TENTATIVELY_UNREACHABLE.
 */
static void
move_legacy_finalizers(PyGC_Head *unreachable, PyGC_Head *finalizers)
{
    PyGC_Head *gc;
    PyGC_Head *next;

    /* March over unreachable.  Move objects with finalizers into
     * `finalizers`.
     */
    for (gc = unreachable->gc.gc_next; gc != unreachable; gc = next) {
        PyObject *op = FROM_GC(gc);

        assert(IS_TENTATIVELY_UNREACHABLE(op));
        next = gc->gc.gc_next;

        if (has_legacy_finalizer(op)) {
            gc_list_move(gc, finalizers);
            _PyGCHead_SET_REFS(gc, GC_REACHABLE);
        }
    }
}

/* A traversal callback for move_legacy_finalizer_reachable. */
static int
visit_move(PyObject *op, PyGC_Head *tolist)
{
    if (PyObject_IS_GC(op)) {
        if (IS_TENTATIVELY_UNREACHABLE(op)) {
            PyGC_Head *gc = AS_GC(op);
            gc_list_move(gc, tolist);
            _PyGCHead_SET_REFS(gc, GC_REACHABLE);
        }
    }
    return 0;
}

/* Move objects that are reachable from finalizers, from the unreachable set
 * into finalizers set.
 */
static void
move_legacy_finalizer_reachable(PyGC_Head *finalizers)
{
    traverseproc traverse;
    PyGC_Head *gc = finalizers->gc.gc_next;
    for (; gc != finalizers; gc = gc->gc.gc_next) {
        /* Note that the finalizers list may grow during this. */
        traverse = Py_TYPE(FROM_GC(gc))->tp_traverse;
        (void) traverse(FROM_GC(gc),
                        (visitproc)visit_move,
                        (void *)finalizers);
    }
}

/* Clear all weakrefs to unreachable objects, and if such a weakref has a
 * callback, invoke it if necessary.  Note that it's possible for such
 * weakrefs to be outside the unreachable set -- indeed, those are precisely
 * the weakrefs whose callbacks must be invoked.  See gc_weakref.txt for
 * overview & some details.  Some weakrefs with callbacks may be reclaimed
 * directly by this routine; the number reclaimed is the return value.  Other
 * weakrefs with callbacks may be moved into the `old` generation.  Objects
 * moved into `old` have gc_refs set to GC_REACHABLE; the objects remaining in
 * unreachable are left at GC_TENTATIVELY_UNREACHABLE.  When this returns,
 * no object in `unreachable` is weakly referenced anymore.
 */
static int
handle_weakrefs(PyGC_Head *unreachable, PyGC_Head *old)
{
    PyGC_Head *gc;
    PyObject *op;               /* generally FROM_GC(gc) */
    PyWeakReference *wr;        /* generally a cast of op */
    PyGC_Head wrcb_to_call;     /* weakrefs with callbacks to call */
    PyGC_Head *next;
    int num_freed = 0;

    gc_list_init(&wrcb_to_call);

    /* Clear all weakrefs to the objects in unreachable.  If such a weakref
     * also has a callback, move it into `wrcb_to_call` if the callback
     * needs to be invoked.  Note that we cannot invoke any callbacks until
     * all weakrefs to unreachable objects are cleared, lest the callback
     * resurrect an unreachable object via a still-active weakref.  We
     * make another pass over wrcb_to_call, invoking callbacks, after this
     * pass completes.
     */
    for (gc = unreachable->gc.gc_next; gc != unreachable; gc = next) {
        PyWeakReference **wrlist;

        op = FROM_GC(gc);
        assert(IS_TENTATIVELY_UNREACHABLE(op));
        next = gc->gc.gc_next;

        if (! PyType_SUPPORTS_WEAKREFS(Py_TYPE(op)))
            continue;

        /* It supports weakrefs.  Does it have any? */
        wrlist = (PyWeakReference **)
                                PyObject_GET_WEAKREFS_LISTPTR(op);

        /* `op` may have some weakrefs.  March over the list, clear
         * all the weakrefs, and move the weakrefs with callbacks
         * that must be called into wrcb_to_call.
         */
        for (wr = *wrlist; wr != NULL; wr = *wrlist) {
            PyGC_Head *wrasgc;                  /* AS_GC(wr) */

            /* _PyWeakref_ClearRef clears the weakref but leaves
             * the callback pointer intact.  Obscure:  it also
             * changes *wrlist.
             */
            assert(wr->wr_object == op);
            _PyWeakref_ClearRef(wr);
            assert(wr->wr_object == Py_None);
            if (wr->wr_callback == NULL)
                continue;                       /* no callback */

    /* Headache time.  `op` is going away, and is weakly referenced by
     * `wr`, which has a callback.  Should the callback be invoked?  If wr
     * is also trash, no:
     *
     * 1. There's no need to call it.  The object and the weakref are
     *    both going away, so it's legitimate to pretend the weakref is
     *    going away first.  The user has to ensure a weakref outlives its
     *    referent if they want a guarantee that the wr callback will get
     *    invoked.
     *
     * 2. It may be catastrophic to call it.  If the callback is also in
     *    cyclic trash (CT), then although the CT is unreachable from
     *    outside the current generation, CT may be reachable from the
     *    callback.  Then the callback could resurrect insane objects.
     *
     * Since the callback is never needed and may be unsafe in this case,
     * wr is simply left in the unreachable set.  Note that because we
     * already called _PyWeakref_ClearRef(wr), its callback will never
     * trigger.
     *
     * OTOH, if wr isn't part of CT, we should invoke the callback:  the
     * weakref outlived the trash.  Note that since wr isn't CT in this
     * case, its callback can't be CT either -- wr acted as an external
     * root to this generation, and therefore its callback did too.  So
     * nothing in CT is reachable from the callback either, so it's hard
     * to imagine how calling it later could create a problem for us.  wr
     * is moved to wrcb_to_call in this case.
     */
            if (IS_TENTATIVELY_UNREACHABLE(wr))
                continue;
            assert(IS_REACHABLE(wr));

            /* Create a new reference so that wr can't go away
             * before we can process it again.
             */
            Py_INCREF(wr);

            /* Move wr to wrcb_to_call, for the next pass. */
            wrasgc = AS_GC(wr);
            assert(wrasgc != next); /* wrasgc is reachable, but
                                       next isn't, so they can't
                                       be the same */
            gc_list_move(wrasgc, &wrcb_to_call);
        }
    }

    /* Invoke the callbacks we decided to honor.  It's safe to invoke them
     * because they can't reference unreachable objects.
     */
    while (! gc_list_is_empty(&wrcb_to_call)) {
        PyObject *temp;
        PyObject *callback;

        gc = wrcb_to_call.gc.gc_next;
        op = FROM_GC(gc);
        assert(IS_REACHABLE(op));
        assert(PyWeakref_Check(op));
        wr = (PyWeakReference *)op;
        callback = wr->wr_callback;
        assert(callback != NULL);

        /* copy-paste of weakrefobject.c's handle_callback() */
        temp = PyObject_CallFunctionObjArgs(callback, wr, NULL);
        if (temp == NULL)
            PyErr_WriteUnraisable(callback);
        else
            Py_DECREF(temp);

        /* Give up the reference we created in the first pass.  When
         * op's refcount hits 0 (which it may or may not do right now),
         * op's tp_dealloc will decref op->wr_callback too.  Note
         * that the refcount probably will hit 0 now, and because this
         * weakref was reachable to begin with, gc didn't already
         * add it to its count of freed objects.  Example:  a reachable
         * weak value dict maps some key to this reachable weakref.
         * The callback removes this key->weakref mapping from the
         * dict, leaving no other references to the weakref (excepting
         * ours).
         */
        Py_DECREF(op);
        if (wrcb_to_call.gc.gc_next == gc) {
            /* object is still alive -- move it */
            gc_list_move(gc, old);
        }
        else
            ++num_freed;
    }

    return num_freed;
}

static void
debug_cycle(const char *msg, PyObject *op)
{
    PySys_FormatStderr("gc: %s <%s %p>\n",
                       msg, Py_TYPE(op)->tp_name, op);
}

/* Handle uncollectable garbage (cycles with tp_del slots, and stuff reachable
 * only from such cycles).
 * If DEBUG_SAVEALL, all objects in finalizers are appended to the module
 * garbage list (a Python list), else only the objects in finalizers with
 * __del__ methods are appended to garbage.  All objects in finalizers are
 * merged into the old list regardless.
 */
static void
handle_legacy_finalizers(PyGC_Head *finalizers, PyGC_Head *old)
{
    PyGC_Head *gc = finalizers->gc.gc_next;

    if (_PyRuntime.gc.garbage == NULL) {
        _PyRuntime.gc.garbage = PyList_New(0);
        if (_PyRuntime.gc.garbage == NULL)
            Py_FatalError("gc couldn't create gc.garbage list");
    }
    for (; gc != finalizers; gc = gc->gc.gc_next) {
        PyObject *op = FROM_GC(gc);

        if ((_PyRuntime.gc.debug & DEBUG_SAVEALL) || has_legacy_finalizer(op)) {
            if (PyList_Append(_PyRuntime.gc.garbage, op) < 0)
                break;
        }
    }

    gc_list_merge(finalizers, old);
}

/* Run first-time finalizers (if any) on all the objects in collectable.
 * Note that this may remove some (or even all) of the objects from the
 * list, due to refcounts falling to 0.
 */
static void
finalize_garbage(PyGC_Head *collectable)
{
    destructor finalize;
    PyGC_Head seen;

    /* While we're going through the loop, `finalize(op)` may cause op, or
     * other objects, to be reclaimed via refcounts falling to zero.  So
     * there's little we can rely on about the structure of the input
     * `collectable` list across iterations.  For safety, we always take the
     * first object in that list and move it to a temporary `seen` list.
     * If objects vanish from the `collectable` and `seen` lists we don't
     * care.
     */
    gc_list_init(&seen);

    while (!gc_list_is_empty(collectable)) {
        PyGC_Head *gc = collectable->gc.gc_next;
        PyObject *op = FROM_GC(gc);
        gc_list_move(gc, &seen);
        if (!_PyGCHead_FINALIZED(gc) &&
                PyType_HasFeature(Py_TYPE(op), Py_TPFLAGS_HAVE_FINALIZE) &&
                (finalize = Py_TYPE(op)->tp_finalize) != NULL) {
            _PyGCHead_SET_FINALIZED(gc, 1);
            Py_INCREF(op);
            finalize(op);
            Py_DECREF(op);
        }
    }
    gc_list_merge(&seen, collectable);
}

/* Walk the collectable list and check that they are really unreachable
   from the outside (some objects could have been resurrected by a
   finalizer). */
static int
check_garbage(PyGC_Head *collectable)
{
    PyGC_Head *gc;
    for (gc = collectable->gc.gc_next; gc != collectable;
         gc = gc->gc.gc_next) {
        _PyGCHead_SET_REFS(gc, Py_REFCNT(FROM_GC(gc)));
        assert(_PyGCHead_REFS(gc) != 0);
    }
    subtract_refs(collectable);
    for (gc = collectable->gc.gc_next; gc != collectable;
         gc = gc->gc.gc_next) {
        assert(_PyGCHead_REFS(gc) >= 0);
        if (_PyGCHead_REFS(gc) != 0)
            return -1;
    }
    return 0;
}

static void
revive_garbage(PyGC_Head *collectable)
{
    PyGC_Head *gc;
    for (gc = collectable->gc.gc_next; gc != collectable;
         gc = gc->gc.gc_next) {
        _PyGCHead_SET_REFS(gc, GC_REACHABLE);
    }
}

/* Break reference cycles by clearing the containers involved.  This is
 * tricky business as the lists can be changing and we don't know which
 * objects may be freed.  It is possible I screwed something up here.
 */
static void
delete_garbage(PyGC_Head *collectable, PyGC_Head *old)
{
    inquiry clear;

    while (!gc_list_is_empty(collectable)) {
        PyGC_Head *gc = collectable->gc.gc_next;
        PyObject *op = FROM_GC(gc);

        if (_PyRuntime.gc.debug & DEBUG_SAVEALL) {
            PyList_Append(_PyRuntime.gc.garbage, op);
        }
        else {
            if ((clear = Py_TYPE(op)->tp_clear) != NULL) {
                Py_INCREF(op);
                clear(op);
                Py_DECREF(op);
            }
        }
        if (collectable->gc.gc_next == gc) {
            /* object is still alive, move it, it may die later */
            gc_list_move(gc, old);
            _PyGCHead_SET_REFS(gc, GC_REACHABLE);
        }
    }
}

/* Clear all free lists
 * All free lists are cleared during the collection of the highest generation.
 * Allocated items in the free list may keep a pymalloc arena occupied.
 * Clearing the free lists may give back memory to the OS earlier.
 */
static void
clear_freelists(void)
{
    (void)PyMethod_ClearFreeList();
    (void)PyFrame_ClearFreeList();
    (void)PyCFunction_ClearFreeList();
    (void)PyTuple_ClearFreeList();
    (void)PyUnicode_ClearFreeList();
    (void)PyFloat_ClearFreeList();
    (void)PyList_ClearFreeList();
    (void)PyDict_ClearFreeList();
    (void)PySet_ClearFreeList();
    (void)PyAsyncGen_ClearFreeLists();
    (void)PyContext_ClearFreeList();
}

/* This is the main function.  Read this to understand how the
 * collection process works. */
static Py_ssize_t
collect(int generation, Py_ssize_t *n_collected, Py_ssize_t *n_uncollectable,
        int nofail)
{
    int i;
    Py_ssize_t m = 0; /* # objects collected */
    Py_ssize_t n = 0; /* # unreachable objects that couldn't be collected */
    PyGC_Head *young; /* the generation we are examining */
    PyGC_Head *old; /* next older generation */
    PyGC_Head unreachable; /* non-problematic unreachable trash */
    PyGC_Head finalizers;  /* objects with, & reachable from, __del__ */
    PyGC_Head *gc;
    _PyTime_t t1 = 0;   /* initialize to prevent a compiler warning */

    struct gc_generation_stats *stats = &_PyRuntime.gc.generation_stats[generation];

    if (_PyRuntime.gc.debug & DEBUG_STATS) {
        PySys_WriteStderr("gc: collecting generation %d...\n",
                          generation);
        PySys_WriteStderr("gc: objects in each generation:");
        for (i = 0; i < NUM_GENERATIONS; i++)
            PySys_FormatStderr(" %zd",
                              gc_list_size(GEN_HEAD(i)));
        PySys_WriteStderr("\ngc: objects in permanent generation: %zd",
                         gc_list_size(&_PyRuntime.gc.permanent_generation.head));
        t1 = _PyTime_GetMonotonicClock();

        PySys_WriteStderr("\n");
    }

    if (PyDTrace_GC_START_ENABLED())
        PyDTrace_GC_START(generation);

    /* update collection and allocation counters */
    if (generation+1 < NUM_GENERATIONS)
        _PyRuntime.gc.generations[generation+1].count += 1;
    for (i = 0; i <= generation; i++)
        _PyRuntime.gc.generations[i].count = 0;

    /* merge younger generations with one we are currently collecting */
    for (i = 0; i < generation; i++) {
        gc_list_merge(GEN_HEAD(i), GEN_HEAD(generation));
    }

    /* handy references */
    young = GEN_HEAD(generation);
    if (generation < NUM_GENERATIONS-1)
        old = GEN_HEAD(generation+1);
    else
        old = young;

    /* Using ob_refcnt and gc_refs, calculate which objects in the
     * container set are reachable from outside the set (i.e., have a
     * refcount greater than 0 when all the references within the
     * set are taken into account).
     */
    update_refs(young);
    subtract_refs(young);

    /* Leave everything reachable from outside young in young, and move
     * everything else (in young) to unreachable.
     * NOTE:  This used to move the reachable objects into a reachable
     * set instead.  But most things usually turn out to be reachable,
     * so it's more efficient to move the unreachable things.
     */
    gc_list_init(&unreachable);
    move_unreachable(young, &unreachable);

    /* Move reachable objects to next generation. */
    if (young != old) {
        if (generation == NUM_GENERATIONS - 2) {
            _PyRuntime.gc.long_lived_pending += gc_list_size(young);
        }
        gc_list_merge(young, old);
    }
    else {
        /* We only untrack dicts in full collections, to avoid quadratic
           dict build-up. See issue #14775. */
        untrack_dicts(young);
        _PyRuntime.gc.long_lived_pending = 0;
        _PyRuntime.gc.long_lived_total = gc_list_size(young);
    }

    /* All objects in unreachable are trash, but objects reachable from
     * legacy finalizers (e.g. tp_del) can't safely be deleted.
     */
    gc_list_init(&finalizers);
    move_legacy_finalizers(&unreachable, &finalizers);
    /* finalizers contains the unreachable objects with a legacy finalizer;
     * unreachable objects reachable *from* those are also uncollectable,
     * and we move those into the finalizers list too.
     */
    move_legacy_finalizer_reachable(&finalizers);

    /* Print debugging information. */
    if (_PyRuntime.gc.debug & DEBUG_COLLECTABLE) {
        for (gc = unreachable.gc.gc_next; gc != &unreachable; gc = gc->gc.gc_next) {
            debug_cycle("collectable", FROM_GC(gc));
        }
    }

    /* Clear weakrefs and invoke callbacks as necessary. */
    m += handle_weakrefs(&unreachable, old);

    /* Call tp_finalize on objects which have one. */
    finalize_garbage(&unreachable);

    if (check_garbage(&unreachable)) {
        revive_garbage(&unreachable);
        gc_list_merge(&unreachable, old);
    }
    else {
        /* Call tp_clear on objects in the unreachable set.  This will cause
         * the reference cycles to be broken.  It may also cause some objects
         * in finalizers to be freed.
         */
        m += gc_list_size(&unreachable);
        delete_garbage(&unreachable, old);
    }

    /* Collect statistics on uncollectable objects found and print
     * debugging information. */
    for (gc = finalizers.gc.gc_next;
         gc != &finalizers;
         gc = gc->gc.gc_next) {
        n++;
        if (_PyRuntime.gc.debug & DEBUG_UNCOLLECTABLE)
            debug_cycle("uncollectable", FROM_GC(gc));
    }
    if (_PyRuntime.gc.debug & DEBUG_STATS) {
        _PyTime_t t2 = _PyTime_GetMonotonicClock();

        if (m == 0 && n == 0)
            PySys_WriteStderr("gc: done");
        else
            PySys_FormatStderr(
                "gc: done, %zd unreachable, %zd uncollectable",
                n+m, n);
        PySys_WriteStderr(", %.4fs elapsed\n",
                          _PyTime_AsSecondsDouble(t2 - t1));
    }

    /* Append instances in the uncollectable set to a Python
     * reachable list of garbage.  The programmer has to deal with
     * this if they insist on creating this type of structure.
     */
    handle_legacy_finalizers(&finalizers, old);

    /* Clear free list only during the collection of the highest
     * generation */
    if (generation == NUM_GENERATIONS-1) {
        clear_freelists();
    }

    if (PyErr_Occurred()) {
        if (nofail) {
            PyErr_Clear();
        }
        else {
            if (gc_str == NULL)
                gc_str = PyUnicode_FromString("garbage collection");
            PyErr_WriteUnraisable(gc_str);
            Py_FatalError("unexpected exception during garbage collection");
        }
    }

    /* Update stats */
    if (n_collected)
        *n_collected = m;
    if (n_uncollectable)
        *n_uncollectable = n;
    stats->collections++;
    stats->collected += m;
    stats->uncollectable += n;

    if (PyDTrace_GC_DONE_ENABLED())
        PyDTrace_GC_DONE(n+m);

    return n+m;
}

/* Invoke progress callbacks to notify clients that garbage collection
 * is starting or stopping
 */
static void
invoke_gc_callback(const char *phase, int generation,
                   Py_ssize_t collected, Py_ssize_t uncollectable)
{
    Py_ssize_t i;
    PyObject *info = NULL;

    /* we may get called very early */
    if (_PyRuntime.gc.callbacks == NULL)
        return;
    /* The local variable cannot be rebound, check it for sanity */
    assert(_PyRuntime.gc.callbacks != NULL && PyList_CheckExact(_PyRuntime.gc.callbacks));
    if (PyList_GET_SIZE(_PyRuntime.gc.callbacks) != 0) {
        info = Py_BuildValue("{sisnsn}",
            "generation", generation,
            "collected", collected,
            "uncollectable", uncollectable);
        if (info == NULL) {
            PyErr_WriteUnraisable(NULL);
            return;
        }
    }
    for (i=0; i<PyList_GET_SIZE(_PyRuntime.gc.callbacks); i++) {
        PyObject *r, *cb = PyList_GET_ITEM(_PyRuntime.gc.callbacks, i);
        Py_INCREF(cb); /* make sure cb doesn't go away */
        r = PyObject_CallFunction(cb, "sO", phase, info);
        if (r == NULL) {
            PyErr_WriteUnraisable(cb);
        }
        else {
            Py_DECREF(r);
        }
        Py_DECREF(cb);
    }
    Py_XDECREF(info);
}

/* Perform garbage collection of a generation and invoke
 * progress callbacks.
 */
static Py_ssize_t
collect_with_callback(int generation)
{
    Py_ssize_t result, collected, uncollectable;
    invoke_gc_callback("start", generation, 0, 0);
    result = collect(generation, &collected, &uncollectable, 0);
    invoke_gc_callback("stop", generation, collected, uncollectable);
    return result;
}

static Py_ssize_t
collect_generations(void)
{
    int i;
    Py_ssize_t n = 0;

    /* Find the oldest generation (highest numbered) where the count
     * exceeds the threshold.  Objects in the that generation and
     * generations younger than it will be collected. */
    for (i = NUM_GENERATIONS-1; i >= 0; i--) {
        if (_PyRuntime.gc.generations[i].count > _PyRuntime.gc.generations[i].threshold) {
            /* Avoid quadratic performance degradation in number
               of tracked objects. See comments at the beginning
               of this file, and issue #4074.
            */
            if (i == NUM_GENERATIONS - 1
                && _PyRuntime.gc.long_lived_pending < _PyRuntime.gc.long_lived_total / 4)
                continue;
            n = collect_with_callback(i);
            break;
        }
    }
    return n;
}

#include "clinic/gcmodule.c.h"

/*[clinic input]
gc.enable

Enable automatic garbage collection.
[clinic start generated code]*/

static PyObject *
gc_enable_impl(PyObject *module)
/*[clinic end generated code: output=45a427e9dce9155c input=81ac4940ca579707]*/
{
    _PyRuntime.gc.enabled = 1;
    Py_RETURN_NONE;
}

/*[clinic input]
gc.disable

Disable automatic garbage collection.
[clinic start generated code]*/

static PyObject *
gc_disable_impl(PyObject *module)
/*[clinic end generated code: output=97d1030f7aa9d279 input=8c2e5a14e800d83b]*/
{
    _PyRuntime.gc.enabled = 0;
    Py_RETURN_NONE;
}

/*[clinic input]
gc.isenabled -> bool

Returns true if automatic garbage collection is enabled.
[clinic start generated code]*/

static int
gc_isenabled_impl(PyObject *module)
/*[clinic end generated code: output=1874298331c49130 input=30005e0422373b31]*/
{
    return _PyRuntime.gc.enabled;
}

/*[clinic input]
gc.collect -> Py_ssize_t

    generation: int(c_default="NUM_GENERATIONS - 1") = 2

Run the garbage collector.

With no arguments, run a full collection.  The optional argument
may be an integer specifying which generation to collect.  A ValueError
is raised if the generation number is invalid.

The number of unreachable objects is returned.
[clinic start generated code]*/

static Py_ssize_t
gc_collect_impl(PyObject *module, int generation)
/*[clinic end generated code: output=b697e633043233c7 input=40720128b682d879]*/
{
    Py_ssize_t n;

    if (generation < 0 || generation >= NUM_GENERATIONS) {
        PyErr_SetString(PyExc_ValueError, "invalid generation");
        return -1;
    }

    if (_PyRuntime.gc.collecting)
        n = 0; /* already collecting, don't do anything */
    else {
        _PyRuntime.gc.collecting = 1;
        n = collect_with_callback(generation);
        _PyRuntime.gc.collecting = 0;
    }

    return n;
}

/*[clinic input]
gc.set_debug

    flags: int
        An integer that can have the following bits turned on:
          DEBUG_STATS - Print statistics during collection.
          DEBUG_COLLECTABLE - Print collectable objects found.
          DEBUG_UNCOLLECTABLE - Print unreachable but uncollectable objects
            found.
          DEBUG_SAVEALL - Save objects to gc.garbage rather than freeing them.
          DEBUG_LEAK - Debug leaking programs (everything but STATS).
    /

Set the garbage collection debugging flags.

Debugging information is written to sys.stderr.
[clinic start generated code]*/

static PyObject *
gc_set_debug_impl(PyObject *module, int flags)
/*[clinic end generated code: output=7c8366575486b228 input=5e5ce15e84fbed15]*/
{
    _PyRuntime.gc.debug = flags;

    Py_RETURN_NONE;
}

/*[clinic input]
gc.get_debug -> int

Get the garbage collection debugging flags.
[clinic start generated code]*/

static int
gc_get_debug_impl(PyObject *module)
/*[clinic end generated code: output=91242f3506cd1e50 input=91a101e1c3b98366]*/
{
    return _PyRuntime.gc.debug;
}

PyDoc_STRVAR(gc_set_thresh__doc__,
"set_threshold(threshold0, [threshold1, threshold2]) -> None\n"
"\n"
"Sets the collection thresholds.  Setting threshold0 to zero disables\n"
"collection.\n");

static PyObject *
gc_set_thresh(PyObject *self, PyObject *args)
{
    int i;
    if (!PyArg_ParseTuple(args, "i|ii:set_threshold",
                          &_PyRuntime.gc.generations[0].threshold,
                          &_PyRuntime.gc.generations[1].threshold,
                          &_PyRuntime.gc.generations[2].threshold))
        return NULL;
    for (i = 2; i < NUM_GENERATIONS; i++) {
        /* generations higher than 2 get the same threshold */
        _PyRuntime.gc.generations[i].threshold = _PyRuntime.gc.generations[2].threshold;
    }

    Py_RETURN_NONE;
}

/*[clinic input]
gc.get_threshold

Return the current collection thresholds.
[clinic start generated code]*/

static PyObject *
gc_get_threshold_impl(PyObject *module)
/*[clinic end generated code: output=7902bc9f41ecbbd8 input=286d79918034d6e6]*/
{
    return Py_BuildValue("(iii)",
                         _PyRuntime.gc.generations[0].threshold,
                         _PyRuntime.gc.generations[1].threshold,
                         _PyRuntime.gc.generations[2].threshold);
}

/*[clinic input]
gc.get_count

Return a three-tuple of the current collection counts.
[clinic start generated code]*/

static PyObject *
gc_get_count_impl(PyObject *module)
/*[clinic end generated code: output=354012e67b16398f input=a392794a08251751]*/
{
    return Py_BuildValue("(iii)",
                         _PyRuntime.gc.generations[0].count,
                         _PyRuntime.gc.generations[1].count,
                         _PyRuntime.gc.generations[2].count);
}

static int
referrersvisit(PyObject* obj, PyObject *objs)
{
    Py_ssize_t i;
    for (i = 0; i < PyTuple_GET_SIZE(objs); i++)
        if (PyTuple_GET_ITEM(objs, i) == obj)
            return 1;
    return 0;
}

static int
gc_referrers_for(PyObject *objs, PyGC_Head *list, PyObject *resultlist)
{
    PyGC_Head *gc;
    PyObject *obj;
    traverseproc traverse;
    for (gc = list->gc.gc_next; gc != list; gc = gc->gc.gc_next) {
        obj = FROM_GC(gc);
        traverse = Py_TYPE(obj)->tp_traverse;
        if (obj == objs || obj == resultlist)
            continue;
        if (traverse(obj, (visitproc)referrersvisit, objs)) {
            if (PyList_Append(resultlist, obj) < 0)
                return 0; /* error */
        }
    }
    return 1; /* no error */
}

PyDoc_STRVAR(gc_get_referrers__doc__,
"get_referrers(*objs) -> list\n\
Return the list of objects that directly refer to any of objs.");

static PyObject *
gc_get_referrers(PyObject *self, PyObject *args)
{
    int i;
    PyObject *result = PyList_New(0);
    if (!result) return NULL;

    for (i = 0; i < NUM_GENERATIONS; i++) {
        if (!(gc_referrers_for(args, GEN_HEAD(i), result))) {
            Py_DECREF(result);
            return NULL;
        }
    }
    return result;
}

/* Append obj to list; return true if error (out of memory), false if OK. */
static int
referentsvisit(PyObject *obj, PyObject *list)
{
    return PyList_Append(list, obj) < 0;
}

PyDoc_STRVAR(gc_get_referents__doc__,
"get_referents(*objs) -> list\n\
Return the list of objects that are directly referred to by objs.");

static PyObject *
gc_get_referents(PyObject *self, PyObject *args)
{
    Py_ssize_t i;
    PyObject *result = PyList_New(0);

    if (result == NULL)
        return NULL;

    for (i = 0; i < PyTuple_GET_SIZE(args); i++) {
        traverseproc traverse;
        PyObject *obj = PyTuple_GET_ITEM(args, i);

        if (! PyObject_IS_GC(obj))
            continue;
        traverse = Py_TYPE(obj)->tp_traverse;
        if (! traverse)
            continue;
        if (traverse(obj, (visitproc)referentsvisit, result)) {
            Py_DECREF(result);
            return NULL;
        }
    }
    return result;
}

/*[clinic input]
gc.get_objects

Return a list of objects tracked by the collector (excluding the list returned).
[clinic start generated code]*/

static PyObject *
gc_get_objects_impl(PyObject *module)
/*[clinic end generated code: output=fcb95d2e23e1f750 input=9439fe8170bf35d8]*/
{
    int i;
    PyObject* result;

    result = PyList_New(0);
    if (result == NULL)
        return NULL;
    for (i = 0; i < NUM_GENERATIONS; i++) {
        if (append_objects(result, GEN_HEAD(i))) {
            Py_DECREF(result);
            return NULL;
        }
    }
    return result;
}

/*[clinic input]
gc.get_stats

Return a list of dictionaries containing per-generation statistics.
[clinic start generated code]*/

static PyObject *
gc_get_stats_impl(PyObject *module)
/*[clinic end generated code: output=a8ab1d8a5d26f3ab input=1ef4ed9d17b1a470]*/
{
    int i;
    PyObject *result;
    struct gc_generation_stats stats[NUM_GENERATIONS], *st;

    /* To get consistent values despite allocations while constructing
       the result list, we use a snapshot of the running stats. */
    for (i = 0; i < NUM_GENERATIONS; i++) {
        stats[i] = _PyRuntime.gc.generation_stats[i];
    }

    result = PyList_New(0);
    if (result == NULL)
        return NULL;

    for (i = 0; i < NUM_GENERATIONS; i++) {
        PyObject *dict;
        st = &stats[i];
        dict = Py_BuildValue("{snsnsn}",
                             "collections", st->collections,
                             "collected", st->collected,
                             "uncollectable", st->uncollectable
                            );
        if (dict == NULL)
            goto error;
        if (PyList_Append(result, dict)) {
            Py_DECREF(dict);
            goto error;
        }
        Py_DECREF(dict);
    }
    return result;

error:
    Py_XDECREF(result);
    return NULL;
}


/*[clinic input]
gc.is_tracked

    obj: object
    /

Returns true if the object is tracked by the garbage collector.

Simple atomic objects will return false.
[clinic start generated code]*/

static PyObject *
gc_is_tracked(PyObject *module, PyObject *obj)
/*[clinic end generated code: output=14f0103423b28e31 input=d83057f170ea2723]*/
{
    PyObject *result;

    if (PyObject_IS_GC(obj) && IS_TRACKED(obj))
        result = Py_True;
    else
        result = Py_False;
    Py_INCREF(result);
    return result;
}

/*[clinic input]
gc.freeze

Freeze all current tracked objects and ignore them for future collections.

This can be used before a POSIX fork() call to make the gc copy-on-write friendly.
Note: collection before a POSIX fork() call may free pages for future allocation
which can cause copy-on-write.
[clinic start generated code]*/

static PyObject *
gc_freeze_impl(PyObject *module)
/*[clinic end generated code: output=502159d9cdc4c139 input=b602b16ac5febbe5]*/
{
    for (int i = 0; i < NUM_GENERATIONS; ++i) {
        gc_list_merge(GEN_HEAD(i), &_PyRuntime.gc.permanent_generation.head);
        _PyRuntime.gc.generations[i].count = 0;
    }
    Py_RETURN_NONE;
}

/*[clinic input]
gc.unfreeze

Unfreeze all objects in the permanent generation.

Put all objects in the permanent generation back into oldest generation.
[clinic start generated code]*/

static PyObject *
gc_unfreeze_impl(PyObject *module)
/*[clinic end generated code: output=1c15f2043b25e169 input=2dd52b170f4cef6c]*/
{
    gc_list_merge(&_PyRuntime.gc.permanent_generation.head, GEN_HEAD(NUM_GENERATIONS-1));
    Py_RETURN_NONE;
}

/*[clinic input]
gc.get_freeze_count -> Py_ssize_t

Return the number of objects in the permanent generation.
[clinic start generated code]*/

static Py_ssize_t
gc_get_freeze_count_impl(PyObject *module)
/*[clinic end generated code: output=61cbd9f43aa032e1 input=45ffbc65cfe2a6ed]*/
{
    return gc_list_size(&_PyRuntime.gc.permanent_generation.head);
}


PyDoc_STRVAR(gc__doc__,
"This module provides access to the garbage collector for reference cycles.\n"
"\n"
"enable() -- Enable automatic garbage collection.\n"
"disable() -- Disable automatic garbage collection.\n"
"isenabled() -- Returns true if automatic collection is enabled.\n"
"collect() -- Do a full collection right now.\n"
"get_count() -- Return the current collection counts.\n"
"get_stats() -- Return list of dictionaries containing per-generation stats.\n"
"set_debug() -- Set debugging flags.\n"
"get_debug() -- Get debugging flags.\n"
"set_threshold() -- Set the collection thresholds.\n"
"get_threshold() -- Return the current the collection thresholds.\n"
"get_objects() -- Return a list of all objects tracked by the collector.\n"
"is_tracked() -- Returns true if a given object is tracked.\n"
"get_referrers() -- Return the list of objects that refer to an object.\n"
"get_referents() -- Return the list of objects that an object refers to.\n"
"freeze() -- Freeze all tracked objects and ignore them for future collections.\n"
"unfreeze() -- Unfreeze all objects in the permanent generation.\n"
"get_freeze_count() -- Return the number of objects in the permanent generation.\n");

static PyMethodDef GcMethods[] = {
    GC_ENABLE_METHODDEF
    GC_DISABLE_METHODDEF
    GC_ISENABLED_METHODDEF
    GC_SET_DEBUG_METHODDEF
    GC_GET_DEBUG_METHODDEF
    GC_GET_COUNT_METHODDEF
    {"set_threshold",  gc_set_thresh, METH_VARARGS, gc_set_thresh__doc__},
    GC_GET_THRESHOLD_METHODDEF
    GC_COLLECT_METHODDEF
    GC_GET_OBJECTS_METHODDEF
    GC_GET_STATS_METHODDEF
    GC_IS_TRACKED_METHODDEF
    {"get_referrers",  gc_get_referrers, METH_VARARGS,
        gc_get_referrers__doc__},
    {"get_referents",  gc_get_referents, METH_VARARGS,
        gc_get_referents__doc__},
    GC_FREEZE_METHODDEF
    GC_UNFREEZE_METHODDEF
    GC_GET_FREEZE_COUNT_METHODDEF
    {NULL,      NULL}           /* Sentinel */
};

static struct PyModuleDef gcmodule = {
    PyModuleDef_HEAD_INIT,
    "gc",              /* m_name */
    gc__doc__,         /* m_doc */
    -1,                /* m_size */
    GcMethods,         /* m_methods */
    NULL,              /* m_reload */
    NULL,              /* m_traverse */
    NULL,              /* m_clear */
    NULL               /* m_free */
};

PyMODINIT_FUNC
PyInit_gc(void)
{
    PyObject *m;

    m = PyModule_Create(&gcmodule);

    if (m == NULL)
        return NULL;

    if (_PyRuntime.gc.garbage == NULL) {
        _PyRuntime.gc.garbage = PyList_New(0);
        if (_PyRuntime.gc.garbage == NULL)
            return NULL;
    }
    Py_INCREF(_PyRuntime.gc.garbage);
    if (PyModule_AddObject(m, "garbage", _PyRuntime.gc.garbage) < 0)
        return NULL;

    if (_PyRuntime.gc.callbacks == NULL) {
        _PyRuntime.gc.callbacks = PyList_New(0);
        if (_PyRuntime.gc.callbacks == NULL)
            return NULL;
    }
    Py_INCREF(_PyRuntime.gc.callbacks);
    if (PyModule_AddObject(m, "callbacks", _PyRuntime.gc.callbacks) < 0)
        return NULL;

#define ADD_INT(NAME) if (PyModule_AddIntConstant(m, #NAME, NAME) < 0) return NULL
    ADD_INT(DEBUG_STATS);
    ADD_INT(DEBUG_COLLECTABLE);
    ADD_INT(DEBUG_UNCOLLECTABLE);
    ADD_INT(DEBUG_SAVEALL);
    ADD_INT(DEBUG_LEAK);
#undef ADD_INT
    return m;
}

/* API to invoke gc.collect() from C */
Py_ssize_t
PyGC_Collect(void)
{
    Py_ssize_t n;

    if (_PyRuntime.gc.collecting)
        n = 0; /* already collecting, don't do anything */
    else {
        PyObject *exc, *value, *tb;
        _PyRuntime.gc.collecting = 1;
        PyErr_Fetch(&exc, &value, &tb);
        n = collect_with_callback(NUM_GENERATIONS - 1);
        PyErr_Restore(exc, value, tb);
        _PyRuntime.gc.collecting = 0;
    }

    return n;
}

Py_ssize_t
_PyGC_CollectIfEnabled(void)
{
    if (!_PyRuntime.gc.enabled)
        return 0;

    return PyGC_Collect();
}

Py_ssize_t
_PyGC_CollectNoFail(void)
{
    Py_ssize_t n;

    /* Ideally, this function is only called on interpreter shutdown,
       and therefore not recursively.  Unfortunately, when there are daemon
       threads, a daemon thread can start a cyclic garbage collection
       during interpreter shutdown (and then never finish it).
       See http://bugs.python.org/issue8713#msg195178 for an example.
       */
    if (_PyRuntime.gc.collecting)
        n = 0;
    else {
        _PyRuntime.gc.collecting = 1;
        n = collect(NUM_GENERATIONS - 1, NULL, NULL, 1);
        _PyRuntime.gc.collecting = 0;
    }
    return n;
}

void
_PyGC_DumpShutdownStats(void)
{
    if (!(_PyRuntime.gc.debug & DEBUG_SAVEALL)
        && _PyRuntime.gc.garbage != NULL && PyList_GET_SIZE(_PyRuntime.gc.garbage) > 0) {
        const char *message;
        if (_PyRuntime.gc.debug & DEBUG_UNCOLLECTABLE)
            message = "gc: %zd uncollectable objects at " \
                "shutdown";
        else
            message = "gc: %zd uncollectable objects at " \
                "shutdown; use gc.set_debug(gc.DEBUG_UNCOLLECTABLE) to list them";
        /* PyErr_WarnFormat does too many things and we are at shutdown,
           the warnings module's dependencies (e.g. linecache) may be gone
           already. */
        if (PyErr_WarnExplicitFormat(PyExc_ResourceWarning, "gc", 0,
                                     "gc", NULL, message,
                                     PyList_GET_SIZE(_PyRuntime.gc.garbage)))
            PyErr_WriteUnraisable(NULL);
        if (_PyRuntime.gc.debug & DEBUG_UNCOLLECTABLE) {
            PyObject *repr = NULL, *bytes = NULL;
            repr = PyObject_Repr(_PyRuntime.gc.garbage);
            if (!repr || !(bytes = PyUnicode_EncodeFSDefault(repr)))
                PyErr_WriteUnraisable(_PyRuntime.gc.garbage);
            else {
                PySys_WriteStderr(
                    "      %s\n",
                    PyBytes_AS_STRING(bytes)
                    );
            }
            Py_XDECREF(repr);
            Py_XDECREF(bytes);
        }
    }
}

void
_PyGC_Fini(void)
{
    Py_CLEAR(_PyRuntime.gc.callbacks);
}

/* for debugging */
void
_PyGC_Dump(PyGC_Head *g)
{
    _PyObject_Dump(FROM_GC(g));
}

/* extension modules might be compiled with GC support so these
   functions must always be available */

#undef PyObject_GC_Track
#undef PyObject_GC_UnTrack
#undef PyObject_GC_Del
#undef _PyObject_GC_Malloc

void
PyObject_GC_Track(void *op)
{
    _PyObject_GC_TRACK(op);
}

void
PyObject_GC_UnTrack(void *op)
{
    /* Obscure:  the Py_TRASHCAN mechanism requires that we be able to
     * call PyObject_GC_UnTrack twice on an object.
     */
    if (IS_TRACKED(op))
        _PyObject_GC_UNTRACK(op);
}

static PyObject *
_PyObject_GC_Alloc(int use_calloc, size_t basicsize)
{
    PyObject *op;
    PyGC_Head *g;
    size_t size;
    if (basicsize > PY_SSIZE_T_MAX - sizeof(PyGC_Head))
        return PyErr_NoMemory();
    size = sizeof(PyGC_Head) + basicsize;
    if (use_calloc)
        g = (PyGC_Head *)PyObject_Calloc(1, size);
    else
        g = (PyGC_Head *)PyObject_Malloc(size);
    if (g == NULL)
        return PyErr_NoMemory();
    g->gc.gc_refs = 0;
    _PyGCHead_SET_REFS(g, GC_UNTRACKED);
    _PyRuntime.gc.generations[0].count++; /* number of allocated GC objects */
    if (_PyRuntime.gc.generations[0].count > _PyRuntime.gc.generations[0].threshold &&
        _PyRuntime.gc.enabled &&
        _PyRuntime.gc.generations[0].threshold &&
        !_PyRuntime.gc.collecting &&
        !PyErr_Occurred()) {
        _PyRuntime.gc.collecting = 1;
        collect_generations();
        _PyRuntime.gc.collecting = 0;
    }
    op = FROM_GC(g);
    return op;
}

PyObject *
_PyObject_GC_Malloc(size_t basicsize)
{
    return _PyObject_GC_Alloc(0, basicsize);
}

PyObject *
_PyObject_GC_Calloc(size_t basicsize)
{
    return _PyObject_GC_Alloc(1, basicsize);
}

PyObject *
_PyObject_GC_New(PyTypeObject *tp)
{
    PyObject *op = _PyObject_GC_Malloc(_PyObject_SIZE(tp));
    if (op != NULL)
        op = PyObject_INIT(op, tp);
    return op;
}

PyVarObject *
_PyObject_GC_NewVar(PyTypeObject *tp, Py_ssize_t nitems)
{
    size_t size;
    PyVarObject *op;

    if (nitems < 0) {
        PyErr_BadInternalCall();
        return NULL;
    }
    size = _PyObject_VAR_SIZE(tp, nitems);
    op = (PyVarObject *) _PyObject_GC_Malloc(size);
    if (op != NULL)
        op = PyObject_INIT_VAR(op, tp, nitems);
    return op;
}

PyVarObject *
_PyObject_GC_Resize(PyVarObject *op, Py_ssize_t nitems)
{
    const size_t basicsize = _PyObject_VAR_SIZE(Py_TYPE(op), nitems);
    PyGC_Head *g = AS_GC(op);
    assert(!IS_TRACKED(op));
    if (basicsize > PY_SSIZE_T_MAX - sizeof(PyGC_Head))
        return (PyVarObject *)PyErr_NoMemory();
    g = (PyGC_Head *)PyObject_REALLOC(g,  sizeof(PyGC_Head) + basicsize);
    if (g == NULL)
        return (PyVarObject *)PyErr_NoMemory();
    op = (PyVarObject *) FROM_GC(g);
    Py_SIZE(op) = nitems;
    return op;
}

void
PyObject_GC_Del(void *op)
{
    PyGC_Head *g = AS_GC(op);
    if (IS_TRACKED(op))
        gc_list_remove(g);
    if (_PyRuntime.gc.generations[0].count > 0) {
        _PyRuntime.gc.generations[0].count--;
    }
    PyObject_FREE(g);
}
