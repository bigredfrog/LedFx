"""macOS main-thread compatibility for pystray's Darwin backend."""

import threading

import Foundation
import objc
import pystray


class _MainThreadCall(Foundation.NSObject):
    """Carry a synchronous callback request to the Cocoa main thread."""

    def initWithCallback_(self, callback):
        self = objc.super(_MainThreadCall, self).init()
        self.callback = callback
        self.event = threading.Event()
        self.result = None
        self.error = None
        return self

    def run(self):
        try:
            self.result = self.callback()
        except BaseException as error:
            self.error = error
        finally:
            self.event.set()


class _MainThreadDispatcher(Foundation.NSObject):
    """Dispatch callbacks through Cocoa's main-thread run loop."""

    def runCall_(self, call):
        call.run()

    def dispatch(self, callback):
        """Run a callback on the main thread and return or re-raise its result."""

        if threading.current_thread() is threading.main_thread():
            return callback()

        call = _MainThreadCall.alloc().initWithCallback_(callback)
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "runCall:", call, False
        )
        call.event.wait()
        if call.error:
            raise call.error
        return call.result


class Icon(pystray.Icon):
    """Dispatch AppKit-mutating pystray operations to the main thread."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._main_thread_dispatcher = _MainThreadDispatcher.alloc().init()

    def _dispatch(self, callback):
        """Synchronously dispatch a pystray backend operation."""

        return self._main_thread_dispatcher.dispatch(callback)

    def _show(self):
        """Show the tray icon on the Cocoa main thread."""

        return self._dispatch(super()._show)

    def _hide(self):
        """Hide the tray icon on the Cocoa main thread."""

        return self._dispatch(super()._hide)

    def _update_icon(self):
        """Update the tray image on the Cocoa main thread."""

        return self._dispatch(super()._update_icon)

    def _update_title(self):
        """Update the tray title on the Cocoa main thread."""

        return self._dispatch(super()._update_title)

    def _update_menu(self):
        """Update the tray menu on the Cocoa main thread."""

        return self._dispatch(super()._update_menu)

    def _stop(self):
        """Stop the tray event loop on the Cocoa main thread."""

        return self._dispatch(super()._stop)
