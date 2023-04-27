# encoding: utf-8
"""Event loop integration for the ZeroMQ-based kernels."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import os
import sys
import platform

import zmq

from distutils.version import LooseVersion as V
from traitlets.config.application import Application
from IPython.utils import io

def _use_appnope():
    """Should we use appnope for dealing with OS X app nap?

    Checks if we are on OS X 10.9 or greater.
    """
    return sys.platform == 'darwin' and V(platform.mac_ver()[0]) >= V('10.9')

def _notify_stream_qt(kernel, stream):

    from IPython.external.qt_for_kernel import QtCore

    if _use_appnope() and kernel._darwin_app_nap:
        from appnope import nope_scope as context
    else:
        from contextlib import contextmanager
        @contextmanager
        def context():
            yield

    def process_stream_events():
        while stream.getsockopt(zmq.EVENTS) & zmq.POLLIN:
            with context():
                kernel.do_one_iteration()

    fd = stream.getsockopt(zmq.FD)
    notifier = QtCore.QSocketNotifier(fd, QtCore.QSocketNotifier.Read, kernel.app)
    notifier.activated.connect(process_stream_events)
    # there may already be unprocessed events waiting.
    # these events will not wake zmq's edge-triggered FD
    # since edge-triggered notification only occurs on new i/o activity.
    # process all the waiting events immediately
    # so we start in a clean state ensuring that any new i/o events will notify.
    # schedule first call on the eventloop as soon as it's running,
    # so we don't block here processing events
    timer = QtCore.QTimer(kernel.app)
    timer.setSingleShot(True)
    timer.timeout.connect(process_stream_events)
    timer.start(0)

# mapping of keys to loop functions
loop_map = {
    'inline': None,
    'nbagg': None,
    'notebook': None,
    'ipympl': None,
    'widget': None,
    None: None,
}

def register_integration(*toolkitnames):
    """Decorator to register an event loop to integrate with the IPython kernel

    The decorator takes names to register the event loop as for the %gui magic.
    You can provide alternative names for the same toolkit.

    The decorated function should take a single argument, the IPython kernel
    instance, arrange for the event loop to call ``kernel.do_one_iteration()``
    at least every ``kernel._poll_interval`` seconds, and start the event loop.

    :mod:`ipykernel.eventloops` provides and registers such functions
    for a few common event loops.
    """
    def decorator(func):
        for name in toolkitnames:
            loop_map[name] = func

        func.exit_hook = lambda kernel: None

        def exit_decorator(exit_func):
            """@func.exit is now a decorator

            to register a function to be called on exit
            """
            func.exit_hook = exit_func

        func.exit = exit_decorator
        return func

    return decorator


def _loop_qt(app):
    """Inner-loop for running the Qt eventloop

    Pulled from guisupport.start_event_loop in IPython < 5.2,
    since IPython 5.2 only checks `get_ipython().active_eventloop` is defined,
    rather than if the eventloop is actually running.
    """
    app._in_event_loop = True
    app.exec_()
    app._in_event_loop = False


@register_integration('qt4')
def loop_qt4(kernel):
    """Start a kernel with PyQt4 event loop integration."""

    from IPython.lib.guisupport import get_app_qt4

    kernel.app = get_app_qt4([" "])
    kernel.app.setQuitOnLastWindowClosed(False)

    for s in kernel.shell_streams:
        _notify_stream_qt(kernel, s)

    _loop_qt(kernel.app)


@loop_qt4.exit
def loop_qt4_exit(kernel):
    kernel.app.exit()


@register_integration('qt', 'qt5')
def loop_qt5(kernel):
    """Start a kernel with PyQt5 event loop integration."""
    os.environ['QT_API'] = 'pyqt5'
    return loop_qt4(kernel)


@loop_qt5.exit
def loop_qt5_exit(kernel):
    kernel.app.exit()


def _loop_wx(app):
    """Inner-loop for running the Wx eventloop

    Pulled from guisupport.start_event_loop in IPython < 5.2,
    since IPython 5.2 only checks `get_ipython().active_eventloop` is defined,
    rather than if the eventloop is actually running.
    """
    app._in_event_loop = True
    app.MainLoop()
    app._in_event_loop = False


@register_integration('wx')
def loop_wx(kernel):
    """Start a kernel with wx event loop support."""

    import wx

    if _use_appnope() and kernel._darwin_app_nap:
        # we don't hook up App Nap contexts for Wx,
        # just disable it outright.
        from appnope import nope
        nope()

    doi = kernel.do_one_iteration
     # Wx uses milliseconds
    poll_interval = int(1000*kernel._poll_interval)

    # We have to put the wx.Timer in a wx.Frame for it to fire properly.
    # We make the Frame hidden when we create it in the main app below.
    class TimerFrame(wx.Frame):
        def __init__(self, func):
            wx.Frame.__init__(self, None, -1)
            self.timer = wx.Timer(self)
            # Units for the timer are in milliseconds
            self.timer.Start(poll_interval)
            self.Bind(wx.EVT_TIMER, self.on_timer)
            self.func = func

        def on_timer(self, event):
            self.func()

    # We need a custom wx.App to create our Frame subclass that has the
    # wx.Timer to drive the ZMQ event loop.
    class IPWxApp(wx.App):
        def OnInit(self):
            self.frame = TimerFrame(doi)
            self.frame.Show(False)
            return True

    # The redirect=False here makes sure that wx doesn't replace
    # sys.stdout/stderr with its own classes.
    kernel.app = IPWxApp(redirect=False)

    # The import of wx on Linux sets the handler for signal.SIGINT
    # to 0.  This is a bug in wx or gtk.  We fix by just setting it
    # back to the Python default.
    import signal
    if not callable(signal.getsignal(signal.SIGINT)):
        signal.signal(signal.SIGINT, signal.default_int_handler)

    _loop_wx(kernel.app)


@loop_wx.exit
def loop_wx_exit(kernel):
    import wx
    wx.Exit()


@register_integration('tk')
def loop_tk(kernel):
    """Start a kernel with the Tk event loop."""

    try:
        from tkinter import Tk  # Py 3
    except ImportError:
        from Tkinter import Tk  # Py 2
    doi = kernel.do_one_iteration
    # Tk uses milliseconds
    poll_interval = int(1000*kernel._poll_interval)
    # For Tkinter, we create a Tk object and call its withdraw method.
    class Timer(object):
        def __init__(self, func):
            self.app = Tk()
            self.app.withdraw()
            self.func = func

        def on_timer(self):
            self.func()
            self.app.after(poll_interval, self.on_timer)

        def start(self):
            self.on_timer()  # Call it once to get things going.
            self.app.mainloop()

    kernel.timer = Timer(doi)
    kernel.timer.start()


@loop_tk.exit
def loop_tk_exit(kernel):
    kernel.timer.app.destroy()


@register_integration('gtk')
def loop_gtk(kernel):
    """Start the kernel, coordinating with the GTK event loop"""
    from .gui.gtkembed import GTKEmbed

    gtk_kernel = GTKEmbed(kernel)
    gtk_kernel.start()
    kernel._gtk = gtk_kernel


@loop_gtk.exit
def loop_gtk_exit(kernel):
    kernel._gtk.stop()


@register_integration('gtk3')
def loop_gtk3(kernel):
    """Start the kernel, coordinating with the GTK event loop"""
    from .gui.gtk3embed import GTKEmbed

    gtk_kernel = GTKEmbed(kernel)
    gtk_kernel.start()
    kernel._gtk = gtk_kernel


@loop_gtk3.exit
def loop_gtk3_exit(kernel):
    kernel._gtk.stop()


@register_integration('osx')
def loop_cocoa(kernel):
    """Start the kernel, coordinating with the Cocoa CFRunLoop event loop
    via the matplotlib MacOSX backend.
    """
    from ._eventloop_macos import mainloop, stop

    real_excepthook = sys.excepthook
    def handle_int(etype, value, tb):
        """don't let KeyboardInterrupts look like crashes"""
        # wake the eventloop when we get a signal
        stop()
        if etype is KeyboardInterrupt:
            io.raw_print("KeyboardInterrupt caught in CFRunLoop")
        else:
            real_excepthook(etype, value, tb)

    while not kernel.shell.exit_now:
        try:
            # double nested try/except, to properly catch KeyboardInterrupt
            # due to pyzmq Issue #130
            try:
                # don't let interrupts during mainloop invoke crash_handler:
                sys.excepthook = handle_int
                mainloop(kernel._poll_interval)
                sys.excepthook = real_excepthook
                kernel.do_one_iteration()
            except:
                raise
        except KeyboardInterrupt:
            # Ctrl-C shouldn't crash the kernel
            io.raw_print("KeyboardInterrupt caught in kernel")
        finally:
            # ensure excepthook is restored
            sys.excepthook = real_excepthook


@loop_cocoa.exit
def loop_cocoa_exit(kernel):
    from ._eventloop_macos import stop
    stop()


@register_integration('asyncio')
def loop_asyncio(kernel):
    '''Start a kernel with asyncio event loop support.'''
    import asyncio
    loop = asyncio.get_event_loop()
    # loop is already running (e.g. tornado 5), nothing left to do
    if loop.is_running():
        return

    def kernel_handler():
        loop.call_soon(kernel.do_one_iteration)
        loop.call_later(kernel._poll_interval, kernel_handler)

    loop.call_soon(kernel_handler)
    while True:
        error = None
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            continue
        except Exception as e:
            error = e
        if hasattr(loop, 'shutdown_asyncgens'):
            loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        if error is not None:
            raise error
        break


@loop_asyncio.exit
def loop_asyncio_exit(kernel):
    """Exit hook for asyncio"""
    import asyncio
    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.call_soon(loop.stop)


def enable_gui(gui, kernel=None):
    """Enable integration with a given GUI"""
    if gui not in loop_map:
        e = "Invalid GUI request %r, valid ones are:%s" % (gui, loop_map.keys())
        raise ValueError(e)
    if kernel is None:
        if Application.initialized():
            kernel = getattr(Application.instance(), 'kernel', None)
        if kernel is None:
            raise RuntimeError("You didn't specify a kernel,"
                " and no IPython Application with a kernel appears to be running."
            )
    loop = loop_map[gui]
    if loop and kernel.eventloop is not None and kernel.eventloop is not loop:
        raise RuntimeError("Cannot activate multiple GUI eventloops")
    kernel.eventloop = loop
