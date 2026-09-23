import sys


def create_icon(name, icon, title):
    if sys.platform == "darwin":
        from ledfx.mac_tray import Icon

        return Icon(name, icon=icon, title=title)

    import pystray

    return pystray.Icon(name, icon=icon, title=title)


def create_menu(*items):
    if sys.platform == "darwin":
        from ledfx.mac_tray import Menu

        return Menu(*items)

    import pystray

    return pystray.Menu(*items)


def create_menu_item(text, callback=None, default=False, enabled=True):
    if sys.platform == "darwin":
        from ledfx.mac_tray import MenuItem

        return MenuItem(
            text, callback, default=default, enabled=enabled
        )

    import pystray

    return pystray.MenuItem(
        text, callback, default=default, enabled=enabled
    )


def menu_separator():
    if sys.platform == "darwin":
        from ledfx.mac_tray import Menu

        return Menu.SEPARATOR

    import pystray

    return pystray.Menu.SEPARATOR
