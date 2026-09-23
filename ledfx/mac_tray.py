import io
import subprocess
import threading

import AppKit
import Foundation
from PIL import Image


class Menu:
    SEPARATOR = object()

    def __init__(self, *items):
        self.items = items


class MenuItem:
    def __init__(self, text, callback=None, default=False, enabled=True):
        self.text = text
        self.callback = callback
        self.default = default
        self.enabled = enabled


class _MainThreadCall:
    def __init__(self, callback):
        self.callback = callback
        self.event = threading.Event()
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = self.callback()
        except BaseException as error:
            self.error = error
        finally:
            self.event.set()


class _MainThreadDispatcher(Foundation.NSObject):
    def runCall_(self, call):
        call.run()


class _MenuDelegate(Foundation.NSObject):
    def initWithIcon_(self, icon):
        self = Foundation.NSObject.init(self)
        self.icon = icon
        return self

    def activateMenuItem_(self, item):
        self.icon._activate_menu_item(item.tag())

class Icon:
    HAS_NOTIFICATION = True

    def __init__(self, name, icon=None, title=None):
        self._name = name
        self._icon = icon
        self._title = title or ""
        self._visible = False
        self._app = AppKit.NSApplication.sharedApplication()
        self._app.setActivationPolicy_(
            AppKit.NSApplicationActivationPolicyAccessory
        )
        self._dispatcher = _MainThreadDispatcher.alloc().init()
        self._delegate = _MenuDelegate.alloc().initWithIcon_(self)
        self._status_bar = AppKit.NSStatusBar.systemStatusBar()
        self._status_item = self._status_bar.statusItemWithLength_(
            AppKit.NSVariableStatusItemLength
        )
        self._menu_callbacks = []
        self._menu = None
        self._icon_image = None

        self._set_icon_image()
        self._status_item.button().setToolTip_(self._title)

    @property
    def menu(self):
        return self._menu

    @menu.setter
    def menu(self, value):
        self._dispatch(lambda: self._set_menu(value))

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        self._dispatch(lambda: self._set_visible(value))

    def run(self, setup=None):
        setup = setup or (lambda icon: setattr(icon, "visible", True))
        self._setup_thread = threading.Thread(
            target=setup, args=(self,), daemon=True
        )
        self._setup_thread.start()
        self._app.run()

    def stop(self):
        self._dispatch(self._stop)

    def notify(self, message, title=None):
        if title is None:
            title = self._title
        subprocess.Popen(
            [
                "osascript",
                "-e",
                'display notification "{}" with title "{}"'.format(
                    message.replace("\\", "\\\\").replace('"', '\\"'),
                    title.replace("\\", "\\\\").replace('"', '\\"'),
                ),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _dispatch(self, callback):
        if threading.current_thread() is threading.main_thread():
            return callback()

        call = _MainThreadCall(callback)
        self._dispatcher.performSelectorOnMainThread_withObject_waitUntilDone_(
            "runCall:", call, False
        )
        call.event.wait()
        if call.error:
            raise call.error
        return call.result

    def _set_icon_image(self):
        if self._icon is None:
            return

        thickness = int(self._status_bar.thickness())
        size = (thickness, thickness)
        source = self._icon
        if source.size != size:
            source = Image.new("RGBA", size)
            source.paste(self._icon.resize(size, Image.Resampling.LANCZOS))

        image_data = io.BytesIO()
        source.save(image_data, "PNG")
        self._icon_image = AppKit.NSImage.alloc().initWithData_(
            Foundation.NSData(image_data.getvalue())
        )
        self._icon_image.setTemplate_(True)
        self._status_item.button().setImage_(self._icon_image)

    def _set_menu(self, menu):
        self._menu = menu
        self._menu_callbacks = []
        native_menu = AppKit.NSMenu.alloc().initWithTitle_(self._name)
        native_menu.setAutoenablesItems_(False)
        for item in menu.items:
            native_menu.addItem_(self._create_menu_item(item))
        self._status_item.setMenu_(native_menu)

    def _create_menu_item(self, item):
        if item is Menu.SEPARATOR:
            return AppKit.NSMenuItem.separatorItem()

        native_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            item.text, b"activateMenuItem:", ""
        )
        native_item.setTarget_(self._delegate)
        native_item.setTag_(len(self._menu_callbacks))
        native_item.setEnabled_(item.enabled)
        if item.default:
            native_item.setAttributedTitle_(
                Foundation.NSAttributedString.alloc().initWithString_attributes_(
                    item.text,
                    {
                        AppKit.NSFontAttributeName: AppKit.NSFont.boldSystemFontOfSize_(
                            AppKit.NSFont.menuFontOfSize_(0).pointSize()
                        )
                    },
                )
            )
        self._menu_callbacks.append(item.callback)
        return native_item

    def _activate_menu_item(self, index):
        callback = self._menu_callbacks[index]
        if callback is not None:
            callback(self)

    def _set_visible(self, value):
        self._visible = value
        self._status_item.setVisible_(value)

    def _stop(self):
        self._status_bar.removeStatusItem_(self._status_item)
        self._app.stop_(None)
        event = AppKit.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
            AppKit.NSApplicationDefined,
            AppKit.NSPoint(0, 0),
            0,
            0.0,
            0,
            None,
            0,
            0,
            0,
        )
        self._app.postEvent_atStart_(event, False)
