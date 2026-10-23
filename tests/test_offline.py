import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import urlopen

import pytest

from memdex.offline import NetworkBlocked, block_network, is_blocked, is_loopback


def test_is_loopback():
    assert is_loopback("127.0.0.1") and is_loopback("localhost") and is_loopback("::1") and is_loopback(None)
    assert not is_loopback("example.com") and not is_loopback("10.0.0.5") and not is_loopback("0.0.0.0")


def test_block_refuses_remote_but_allows_loopback():
    block_network()
    assert is_blocked()
    with pytest.raises(NetworkBlocked):
        socket.getaddrinfo("example.com", 443)
    with pytest.raises(NetworkBlocked):
        socket.create_connection(("93.184.216.34", 80), timeout=1)

    class Ping(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Ping)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        assert urlopen(f"http://127.0.0.1:{srv.server_address[1]}/", timeout=5).read() == b"ok"
    finally:
        srv.shutdown()
        srv.server_close()
