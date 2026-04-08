#!/usr/bin/env python3
"""
Robust local CONNECT proxy that forwards through the upstream HTTP proxy.
Handles concurrent connections and auto-recovers from errors.
"""
import base64, http.server, select, socket, socketserver, sys, threading, traceback

UPSTREAM_PROXY = ("192.168.189.225", 1080)
UPSTREAM_AUTH = base64.b64encode(b"user:passWorD").decode()
LOCAL_PORT = 18080


class ConnectProxyHandler(http.server.BaseHTTPRequestHandler):
    timeout = 300

    def do_CONNECT(self):
        host, port = self.path.split(":")
        port = int(port)
        upstream = None
        try:
            upstream = socket.create_connection(UPSTREAM_PROXY, timeout=15)
            req = (
                f"CONNECT {host}:{port} HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                f"Proxy-Authorization: Basic {UPSTREAM_AUTH}\r\n"
                f"User-Agent: curl/7.88.1\r\n"
                f"Proxy-Connection: Keep-Alive\r\n"
                f"\r\n"
            )
            upstream.sendall(req.encode())

            resp = b""
            while b"\r\n\r\n" not in resp:
                chunk = upstream.recv(4096)
                if not chunk:
                    self.send_error(502, "Upstream closed")
                    return
                resp += chunk

            status_line = resp.split(b"\r\n")[0].decode()
            if "200" not in status_line:
                self.send_error(502, f"Upstream refused: {status_line}")
                return

            self.send_response(200, "Connection established")
            self.end_headers()

            client_sock = self.connection
            sockets = [client_sock, upstream]
            while True:
                readable, _, errored = select.select(sockets, [], sockets, 120)
                if errored:
                    break
                if not readable:
                    break
                for s in readable:
                    data = s.recv(65536)
                    if not data:
                        return
                    target = upstream if s is client_sock else client_sock
                    target.sendall(data)
        except Exception:
            pass
        finally:
            if upstream:
                try:
                    upstream.close()
                except Exception:
                    pass

    def log_message(self, format, *args):
        pass


class ThreadedProxy(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128


if __name__ == "__main__":
    server = ThreadedProxy(("127.0.0.1", LOCAL_PORT), ConnectProxyHandler)
    print(f"Local CONNECT proxy on 127.0.0.1:{LOCAL_PORT}", flush=True)
    server.serve_forever()
