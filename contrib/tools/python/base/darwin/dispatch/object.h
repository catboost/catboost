/*
 * Copyright (c) 2008-2012 Apple Inc. All rights reserved.
 *
 * @APPLE_APACHE_LICENSE_HEADER_START@
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @APPLE_APACHE_LICENSE_HEADER_END@
 */

#ifndef __DISPATCH_OBJECT__
#define __DISPATCH_OBJECT__

#ifndef __DISPATCH_INDIRECT__
#error "Please #include <dispatch/dispatch.h> instead of this file directly."
#include <dispatch/base.h> // for HeaderDoc
#endif

/*!
 * @typedef dispatch_object_t
 *
 * @abstract
 * Abstract base type for all dispatch objects.
 * The details of the type definition are language-specific.
 *
 * @discussion
 * Dispatch objects are reference counted via calls to dispatch_retain() and
 * dispatch_release().
 */

#if OS_OBJECT_USE_OBJC
/*
 * By default, dispatch objects are declared as Objective-C types when building
 * with an Objective-C compiler. This allows them to participate in ARC, in RR
 * management by the Blocks runtime and in leaks checking by the static
 * analyzer, and enables them to be added to Cocoa collections.
 * See <os/object.h> for details.
 */
OS_OBJECT_DECL(dispatch_object);
#define DISPATCH_DECL(name) OS_OBJECT_DECL_SUBCLASS(name, dispatch_object)
#define DISPATCH_GLOBAL_OBJECT(type, object) ((OS_OBJECT_BRIDGE type)&(object))
#define DISPATCH_RETURNS_RETAINED OS_OBJECT_RETURNS_RETAINED
DISPATCH_INLINE DISPATCH_ALWAYS_INLINE DISPATCH_NONNULL_ALL DISPATCH_NOTHROW
void
_dispatch_object_validate(dispatch_object_t object) {
	void *isa = *(void* volatile*)(OS_OBJECT_BRIDGE void*)object;
	(void)isa;
}
#elif defined(__cplusplus)
/*
 * Dispatch objects are NOT C++ objects. Nevertheless, we can at least keep C++
 * aware of type compatibility.
 */
typedef struct dispatch_object_s {
private:
	dispatch_object_s();
	~dispatch_object_s();
	dispatch_object_s(const dispatch_object_s &);
	void operator=(const dispatch_object_s &);
} *dispatch_object_t;
#define DISPATCH_DECL(name) \
		typedef struct name##_s : public dispatch_object_s {} *name##_t
#define DISPATCH_GLOBAL_OBJECT(type, object) (&(object))
#define DISPATCH_RETURNS_RETAINED
#else /* Plain C */
typedef union {
	struct _os_object_s *_os_obj;
	struct dispatch_object_s *_do;
	struct dispatch_continuation_s *_dc;
	struct dispatch_queue_s *_dq;
	struct dispatch_queue_attr_s *_dqa;
	struct dispatch_group_s *_dg;
	struct dispatch_source_s *_ds;
	struct dispatch_mach_s *_dm;
	struct dispatch_mach_msg_s *_dmsg;
	struct dispatch_timer_aggregate_s *_dta;
	struct dispatch_source_attr_s *_dsa;
	struct dispatch_semaphore_s *_dsema;
	struct dispatch_data_s *_ddata;
	struct dispatch_io_s *_dchannel;
	struct dispatch_operation_s *_doperation;
	struct dispatch_disk_s *_ddisk;
} dispatch_object_t __attribute__((__transparent_union__));
/*! @parseOnly */
#define DISPATCH_DECL(name) typedef struct name##_s *name##_t
/*! @parseOnly */
#define DISPATCH_GLOBAL_OBJECT(t, x) (&(x))
/*! @parseOnly */
#define DISPATCH_RETURNS_RETAINED
#endif

/*!
 * @typedef dispatch_block_t
 *
 * @abstract
 * The type of blocks submitted to dispatch queues, which take no arguments
 * and have no return value.
 *
 * @discussion
 * When not building with Objective-C ARC, a block object allocated on or
 * copied to the heap must be released with a -[release] message or the
 * Block_release() function.
 *
 * The declaration of a block literal allocates storage on the stack.
 * Therefore, this is an invalid construct:
 * <code>
 * dispatch_block_t block;
 * if (x) {
 *     block = ^{ printf("true\n"); };
 * } else {
 *     block = ^{ printf("false\n"); };
 * }
 * block(); // unsafe!!!
 * </code>
 *
 * What is happening behind the scenes:
 * <code>
 * if (x) {
 *     struct Block __tmp_1 = ...; // setup details
 *     block = &__tmp_1;
 * } else {
 *     struct Block __tmp_2 = ...; // setup details
 *     block = &__tmp_2;
 * }
 * </code>
 *
 * As the example demonstrates, the address of a stack variable is escaping the
 * scope in which it is allocated. That is a classic C bug.
 *
 * Instead, the block literal must be copied to the heap with the Block_copy()
 * function or by sending it a -[copy] message.
 */
typedef void (*dispatch_block_t)(void);

__BEGIN_DECLS

/*!
 * @function dispatch_retain
 *
 * @abstract
 * Increment the reference count of a dispatch object.
 *
 * @discussion
 * Calls to dispatch_retain() must be balanced with calls to
 * dispatch_release().
 *
 * @param object
 * The object to retain.
 * The result of passing NULL in this parameter is undefined.
 */
__OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_4_0)
DISPATCH_EXPORT DISPATCH_NONNULL_ALL DISPATCH_NOTHROW
void
dispatch_retain(dispatch_object_t object);
#if OS_OBJECT_USE_OBJC_RETAIN_RELEASE
#undef dispatch_retain
#define dispatch_retain(object) ({ dispatch_object_t _o = (object); \
		_dispatch_object_validate(_o); (void)[_o retain]; })
#endif

/*!
 * @function dispatch_release
 *
 * @abstract
 * Decrement the reference count of a dispatch object.
 *
 * @discussion
 * A dispatch object is asynchronously deallocated once all references are
 * released (i.e. the reference count becomes zero). The system does not
 * guarantee that a given client is the last or only reference to a given
 * object.
 *
 * @param object
 * The object to release.
 * The result of passing NULL in this parameter is undefined.
 */
__OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_4_0)
DISPATCH_EXPORT DISPATCH_NONNULL_ALL DISPATCH_NOTHROW
void
dispatch_release(dispatch_object_t object);
#if OS_OBJECT_USE_OBJC_RETAIN_RELEASE
#undef dispatch_release
#define dispatch_release(object) ({ dispatch_object_t _o = (object); \
		_dispatch_object_validate(_o); [_o release]; })
#endif

/*!
 * @function dispatch_get_context
 *
 * @abstract
 * Returns the application defined context of the object.
 *
 * @param object
 * The result of passing NULL in this parameter is undefined.
 *
 * @result
 * The context of the object; may be NULL.
 */
__OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_4_0)
DISPATCH_EXPORT DISPATCH_NONNULL_ALL DISPATCH_PURE DISPATCH_WARN_RESULT
DISPATCH_NOTHROW
void *
dispatch_get_context(dispatch_object_t object);

/*!
 * @function dispatch_set_context
 *
 * @abstract
 * Associates an application defined context with the object.
 *
 * @param object
 * The result of passing NULL in this parameter is undefined.
 *
 * @param context
 * The new client defined context for the object. This may be NULL.
 *
 */
__OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_4_0)
DISPATCH_EXPORT DISPATCH_NOTHROW //DISPATCH_NONNULL1
void
dispatch_set_context(dispatch_object_t object, void *context);

/*!
 * @function dispatch_set_finalizer_f
 *
 * @abstract
 * Set the finalizer function for a dispatch object.
 *
 * @param object
 * The dispatch object to modify.
 * The result of passing NULL in this parameter is undefined.
 *
 * @param finalizer
 * The finalizer function pointer.
 *
 * @discussion
 * A dispatch object's finalizer will be invoked on the object's target queue
 * after all references to the object have been released. This finalizer may be
 * used by the application to release any resources associated with the object,
 * such as freeing the object's context.
 * The context parameter passed to the finalizer function is the current
 * context of the dispatch object at the time the finalizer call is made.
 */
__OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_4_0)
DISPATCH_EXPORT DISPATCH_NOTHROW //DISPATCH_NONNULL1
void
dispatch_set_finalizer_f(dispatch_object_t object,
		dispatch_function_t finalizer);

/*!
 * @function dispatch_suspend
 *
 * @abstract
 * Suspends the invocation of blocks on a dispatch object.
 *
 * @discussion
 * A suspended object will not invoke any blocks associated with it. The
 * suspension of an object will occur after any running block associated with
 * the object completes.
 *
 * Calls to dispatch_suspend() must be balanced with calls
 * to dispatch_resume().
 *
 * @param object
 * The object to be suspended.
 * The result of passing NULL in this parameter is undefined.
 */
__OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_4_0)
DISPATCH_EXPORT DISPATCH_NONNULL_ALL DISPATCH_NOTHROW
void
dispatch_suspend(dispatch_object_t object);

/*!
 * @function dispatch_resume
 *
 * @abstract
 * Resumes the invocation of blocks on a dispatch object.
 *
 * @param object
 * The object to be resumed.
 * The result of passing NULL in this parameter is undefined.
 */
__OSX_AVAILABLE_STARTING(__MAC_10_6,__IPHONE_4_0)
DISPATCH_EXPORT DISPATCH_NONNULL_ALL DISPATCH_NOTHROW
void
dispatch_resume(dispatch_object_t object);

/*!
 * @function dispatch_wait
 *
 * @abstract
 * Wait synchronously for an object or until the specified timeout has elapsed.
 *
 * @discussion
 * Type-generic macro that maps to dispatch_block_wait, dispatch_group_wait or
 * dispatch_semaphore_wait, depending on the type of the first argument.
 * See documentation for these functions for more details.
 * This function is unavailable for any other object type.
 *
 * @param object
 * The object to wait on.
 * The result of passing NULL in this parameter is undefined.
 *
 * @param timeout
 * When to timeout (see dispatch_time). As a convenience, there are the
 * DISPATCH_TIME_NOW and DISPATCH_TIME_FOREVER constants.
 *
 * @result
 * Returns zero on success or non-zero on error (i.e. timed out).
 */
DISPATCH_UNAVAILABLE
DISPATCH_EXPORT DISPATCH_NONNULL1 DISPATCH_NOTHROW
long
dispatch_wait(void *object, dispatch_time_t timeout);
#if __has_extension(c_generic_selections)
#define dispatch_wait(object, timeout) \
		_Generic((object), \
			dispatch_block_t:dispatch_block_wait, \
			dispatch_group_t:dispatch_group_wait, \
			dispatch_semaphore_t:dispatch_semaphore_wait \
		)((object),(timeout))
#endif

/*!
 * @function dispatch_notify
 *
 * @abstract
 * Schedule a notification block to be submitted to a queue when the execution
 * of a specified object has completed.
 *
 * @discussion
 * Type-generic macro that maps to dispatch_block_notify or
 * dispatch_group_notify, depending on the type of the first argument.
 * See documentation for these functions for more details.
 * This function is unavailable for any other object type.
 *
 * @param object
 * The object to observe.
 * The result of passing NULL in this parameter is undefined.
 *
 * @param queue
 * The queue to which the supplied notification block will be submitted when
 * the observed object completes.
 *
 * @param notification_block
 * The block to submit when the observed object completes.
 */
DISPATCH_UNAVAILABLE
DISPATCH_EXPORT DISPATCH_NONNULL_ALL DISPATCH_NOTHROW
void
dispatch_notify(void *object, dispatch_object_t queue,
		dispatch_block_t notification_block);
#if __has_extension(c_generic_selections)
#define dispatch_notify(object, queue, notification_block) \
		_Generic((object), \
			dispatch_block_t:dispatch_block_notify, \
			dispatch_group_t:dispatch_group_notify \
		)((object),(queue), (notification_block))
#endif

/*!
 * @function dispatch_cancel
 *
 * @abstract
 * Cancel the specified object.
 *
 * @discussion
 * Type-generic macro that maps to dispatch_block_cancel or
 * dispatch_source_cancel, depending on the type of the first argument.
 * See documentation for these functions for more details.
 * This function is unavailable for any other object type.
 *
 * @param object
 * The object to cancel.
 * The result of passing NULL in this parameter is undefined.
 */
DISPATCH_UNAVAILABLE
DISPATCH_EXPORT DISPATCH_NONNULL_ALL DISPATCH_NOTHROW
void
dispatch_cancel(void *object);
#if __has_extension(c_generic_selections)
#define dispatch_cancel(object) \
		_Generic((object), \
			dispatch_block_t:dispatch_block_cancel, \
			dispatch_source_t:dispatch_source_cancel \
		)((object))
#endif

/*!
 * @function dispatch_testcancel
 *
 * @abstract
 * Test whether the specified object has been canceled
 *
 * @discussion
 * Type-generic macro that maps to dispatch_block_testcancel or
 * dispatch_source_testcancel, depending on the type of the first argument.
 * See documentation for these functions for more details.
 * This function is unavailable for any other object type.
 *
 * @param object
 * The object to test.
 * The result of passing NULL in this parameter is undefined.
 *
 * @result
 * Non-zero if canceled and zero if not canceled.
 */
DISPATCH_UNAVAILABLE
DISPATCH_EXPORT DISPATCH_NONNULL_ALL DISPATCH_WARN_RESULT DISPATCH_PURE
DISPATCH_NOTHROW
long
dispatch_testcancel(void *object);
#if __has_extension(c_generic_selections)
#define dispatch_testcancel(object) \
		_Generic((object), \
			dispatch_block_t:dispatch_block_testcancel, \
			dispatch_source_t:dispatch_source_testcancel \
		)((object))
#endif

/*!
 * @function dispatch_debug
 *
 * @abstract
 * Programmatically log debug information about a dispatch object.
 *
 * @discussion
 * Programmatically log debug information about a dispatch object. By default,
 * the log output is sent to syslog at notice level. In the debug version of
 * the library, the log output is sent to a file in /var/tmp.
 * The log output destination can be configured via the LIBDISPATCH_LOG
 * environment variable, valid values are: YES, NO, syslog, stderr, file.
 *
 * This function is deprecated and will be removed in a future release.
 * Objective-C callers may use -debugDescription instead.
 *
 * @param object
 * The object to introspect.
 *
 * @param message
 * The message to log above and beyond the introspection.
 */
__OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_6,__MAC_10_9,__IPHONE_4_0,__IPHONE_6_0)
DISPATCH_EXPORT DISPATCH_NONNULL2 DISPATCH_NOTHROW
__attribute__((__format__(printf,2,3)))
void
dispatch_debug(dispatch_object_t object, const char *message, ...);

__OSX_AVAILABLE_BUT_DEPRECATED(__MAC_10_6,__MAC_10_9,__IPHONE_4_0,__IPHONE_6_0)
DISPATCH_EXPORT DISPATCH_NONNULL2 DISPATCH_NOTHROW
__attribute__((__format__(printf,2,0)))
void
dispatch_debugv(dispatch_object_t object, const char *message, va_list ap);

__END_DECLS

#endif
