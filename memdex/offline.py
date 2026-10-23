"""Make 'no network at runtime' a property of the code, not a promise.

block_network() makes any Python-level socket connection or DNS lookup to a non-loopback address
raise. Every command except `fetch-model` runs with it on, so an accidental call (a library phoning
home, a model auto-download) fails loudly instead of happening silently.

Loopback (127.x, ::1, localhost) stays open: the web UI serves on 127.0.0.1 and that is not "the network".
"""

from __future__ import annotations

import socket


class NetworkBlocked(RuntimeError):
    pass


_blocked = False
_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex
_orig_create_connection = socket.create_connection
_orig_getaddrinfo = socket.getaddrinfo


def is_loopback(host: object) -> bool:
    if host is None:
        return True  # getaddrinfo(None, port) = the local machine
    h = str(host).strip("[]").lower()
    return h in ("localhost", "::1", "") or h.startswith("127.")


def _host_of(address: object) -> object:
    return address[0] if isinstance(address, tuple) and address else address


def _refuse(host: object) -> None:
    raise NetworkBlocked(f"MemDex runs offline: a network call to {host!r} was attempted and blocked")


def _connect(self, address):
    host = _host_of(address)
    if not is_loopback(host):
        _refuse(host)
    return _orig_connect(self, address)


def _connect_ex(self, address):
    host = _host_of(address)
    if not is_loopback(host):
        _refuse(host)
    return _orig_connect_ex(self, address)


def _create_connection(address, *args, **kwargs):
    host = _host_of(address)
    if not is_loopback(host):
        _refuse(host)
    return _orig_create_connection(address, *args, **kwargs)


def _getaddrinfo(host, *args, **kwargs):
    if not is_loopback(host):
        _refuse(host)
    return _orig_getaddrinfo(host, *args, **kwargs)


def block_network() -> None:
    global _blocked
    if _blocked:
        return
    socket.socket.connect = _connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _connect_ex  # type: ignore[method-assign]
    socket.create_connection = _create_connection  # type: ignore[assignment]
    socket.getaddrinfo = _getaddrinfo  # type: ignore[assignment]
    _blocked = True


def is_blocked() -> bool:
    return _blocked
