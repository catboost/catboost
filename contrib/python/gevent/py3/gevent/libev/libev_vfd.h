#ifdef _WIN32
/* see discussion in the libuv directory: this is a SOCKET which is a
   HANDLE which is a PVOID (even though they're really small ints),
   and CPython and PyPy return that SOCKET cast to an int from
   fileno()
*/
typedef intptr_t vfd_socket_t;
#define vfd_socket_object PyLong_FromLongLong

#ifdef LIBEV_EMBED
/*
 * If libev on win32 is embedded, then we can use an
 * arbitrary mapping between integer fds and OS
 * handles. Then by defining special macros libev
 * will use our functions.
 */

#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <windows.h>

typedef struct vfd_entry_t
{
	vfd_socket_t handle; /* OS handle, i.e. SOCKET */
	int count; /* Reference count, 0 if free */
	int next; /* Next free fd, -1 if last */
} vfd_entry;

#define VFD_INCREMENT 128
static int vfd_num = 0; /* num allocated fds */
static int vfd_max = 0; /* max allocated fds */
static int vfd_next = -1; /* next free fd for reuse */
static PyObject* vfd_map = NULL; /* map OS handle -> virtual fd */
static vfd_entry* vfd_entries = NULL; /* list of virtual fd entries */

#ifdef WITH_THREAD
static CRITICAL_SECTION* volatile vfd_lock = NULL;
static CRITICAL_SECTION* vfd_make_lock()
{
	if (vfd_lock == NULL) {
		/* must use malloc and not PyMem_Malloc here */
		CRITICAL_SECTION* lock = malloc(sizeof(CRITICAL_SECTION));
		InitializeCriticalSection(lock);
		if (InterlockedCompareExchangePointer(&vfd_lock, lock, NULL) != NULL) {
			/* another thread initialized lock first */
			DeleteCriticalSection(lock);
			free(lock);
		}
	}
	return vfd_lock;
}
#define VFD_LOCK_ENTER  EnterCriticalSection(vfd_make_lock())
#define VFD_LOCK_LEAVE  LeaveCriticalSection(vfd_lock)
#define VFD_GIL_DECLARE PyGILState_STATE ___save
#define VFD_GIL_ENSURE  ___save = PyGILState_Ensure()
#define VFD_GIL_RELEASE PyGILState_Release(___save)
#else /* ! WITH_THREAD */
#define VFD_LOCK_ENTER
#define VFD_LOCK_LEAVE
#define VFD_GIL_DECLARE
#define VFD_GIL_ENSURE
#define VFD_GIL_RELEASE
#endif /*_WITH_THREAD */

/*
 * Given a virtual fd returns an OS handle or -1
 * This function is speed critical, so it cannot use GIL
 */
static vfd_socket_t vfd_get(int fd)
{
	vfd_socket_t handle = -1;
	VFD_LOCK_ENTER;
	if (vfd_entries != NULL && fd >= 0 && fd < vfd_num)
		handle = vfd_entries[fd].handle;
	VFD_LOCK_LEAVE;
	return handle;
}

#define EV_FD_TO_WIN32_HANDLE(fd) vfd_get((fd))

/*
 * Given an OS handle finds or allocates a virtual fd
 * Returns -1 on failure and sets Python exception if pyexc is non-zero
 */
static int vfd_open_(vfd_socket_t handle, int pyexc)
{
	VFD_GIL_DECLARE;
	int fd = -1;
	unsigned long arg;
	PyObject* key = NULL;
	PyObject* value;

	if (!pyexc) {
		VFD_GIL_ENSURE;
	}
	if (ioctlsocket(handle, FIONREAD, &arg) != 0) {
		if (pyexc)
			PyErr_Format(PyExc_IOError,
#ifdef _WIN64
				"%lld is not a socket (files are not supported)",
#else
				"%ld is not a socket (files are not supported)",
#endif
				handle);
		goto done;
	}
	if (vfd_map == NULL) {
		vfd_map = PyDict_New();
		if (vfd_map == NULL)
			goto done;
	}
	key = vfd_socket_object(handle);
	/* check if it's already in the dict */
	value = PyDict_GetItem(vfd_map, key);
	if (value != NULL) {
		/* is it safe to use PyInt_AS_LONG(value) here? */
		fd = PyInt_AsLong(value);
		if (fd >= 0) {
			++vfd_entries[fd].count;
			goto done;
		}
	}
	/* use the free entry, if available */
	if (vfd_next >= 0) {
		fd = vfd_next;
		vfd_next = vfd_entries[fd].next;
		VFD_LOCK_ENTER;
		goto allocated;
	}
	/* check if it would be out of bounds */
	if (vfd_num >= FD_SETSIZE) {
		/* libev's select doesn't support more that FD_SETSIZE fds */
		if (pyexc)
			PyErr_Format(PyExc_IOError, "cannot watch more than %d sockets", (int)FD_SETSIZE);
		goto done;
	}
	/* allocate more space if needed */
	VFD_LOCK_ENTER;
	if (vfd_num >= vfd_max) {
		int newsize = vfd_max + VFD_INCREMENT;
		vfd_entry* entries = PyMem_Realloc(vfd_entries, sizeof(vfd_entry) * newsize);
		if (entries == NULL) {
			VFD_LOCK_LEAVE;
			if (pyexc)
				PyErr_NoMemory();
			goto done;
		}
		vfd_entries = entries;
		vfd_max = newsize;
	}
	fd = vfd_num++;
allocated:
	/* vfd_lock must be acquired when entering here */
	vfd_entries[fd].handle = handle;
	vfd_entries[fd].count = 1;
	VFD_LOCK_LEAVE;
	value = PyInt_FromLong(fd);
	PyDict_SetItem(vfd_map, key, value);
	Py_DECREF(value);
done:
	Py_XDECREF(key);
	if (!pyexc) {
		VFD_GIL_RELEASE;
	}
	return fd;
}

#define vfd_open(fd) vfd_open_((fd), 1)
#define EV_WIN32_HANDLE_TO_FD(handle) vfd_open_((handle), 0)

static void vfd_free_(int fd, int needclose)
{
	VFD_GIL_DECLARE;
	PyObject* key;

	if (needclose) {
		VFD_GIL_ENSURE;
	}
	if (fd < 0 || fd >= vfd_num)
		goto done; /* out of bounds */
	if (vfd_entries[fd].count <= 0)
		goto done; /* free entry, ignore */
	if (!--vfd_entries[fd].count) {
		/* fd has just been freed */
		vfd_socket_t handle = vfd_entries[fd].handle;
		vfd_entries[fd].handle = -1;
		vfd_entries[fd].next = vfd_next;
		vfd_next = fd;
		if (needclose)
			closesocket(handle);
		/* vfd_map is assumed to be != NULL */
		key = vfd_socket_object(handle);
		PyDict_DelItem(vfd_map, key);
		Py_DECREF(key);
	}
done:
	if (needclose) {
		VFD_GIL_RELEASE;
	}
}

#define vfd_free(fd) vfd_free_((fd), 0)
#define EV_WIN32_CLOSE_FD(fd) vfd_free_((fd), 1)

#else /* !LIBEV_EMBED */
/*
 * If libev on win32 is not embedded in gevent, then
 * the only way to map vfds is to use the default of
 * using C runtime fds in libev. Note that it will leak
 * fds, because there's no way of closing them safely
 */
#define vfd_get(fd) _get_osfhandle((fd))
#define vfd_open(fd) _open_osfhandle((fd), 0)
#define vfd_free(fd)
#endif /* LIBEV_EMBED */

#else /* !_WIN32 */
/*
 * On non-win32 platforms vfd_* are noop macros
 */
typedef int vfd_socket_t;
#define vfd_get(fd) (fd)
#define vfd_open(fd) (fd)
#define vfd_free(fd)
#endif /* _WIN32 */
