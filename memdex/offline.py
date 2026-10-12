"""Make 'no network at runtime' a property of the code, not a promise.

block_network() makes any Python-level socket connection or DNS lookup raise. Every command except
`fetch-model` runs with it on, so an accidental call (a library phoning home, a model auto-download)
fails loudly instead of happening silently.
"""

from __future__ import annotations

import socket


class NetworkBlocked(RuntimeError):
    pass


_blocked = False


def _refuse(*args, **kwargs):
    raise NetworkBlocked("MemDex runs offline: a network call was attempted and blocked")


def block_network() -> None:
    global _blocked
    if _blocked:
        return
    socket.socket.connect = _refuse  # type: ignore[method-assign]
    socket.socket.connect_ex = _refuse  # type: ignore[method-assign]
    socket.create_connection = _refuse  # type: ignore[assignment]
    socket.getaddrinfo = _refuse  # type: ignore[assignment]
    _blocked = True
