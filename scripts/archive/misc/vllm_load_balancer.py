#!/usr/bin/env python3
"""
Simple round-robin load balancer for multiple vLLM servers.
Listens on port 8200 and distributes requests to ports 8100-8107.
"""
import http.server
import itertools
import socketserver
import threading
import urllib.request
import urllib.error

BACKENDS = [f"http://localhost:{8100 + i}" for i in range(8)]
LISTEN_PORT = 8200
_backend_cycle = itertools.cycle(BACKENDS)
_lock = threading.Lock()


class LBHandler(http.server.BaseHTTPRequestHandler):
    def _get_backend(self):
        with _lock:
            return next(_backend_cycle)

    def _proxy(self):
        backend = self._get_backend()
        url = f"{backend}{self.path}"

        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # Forward request
        req = urllib.request.Request(
            url, data=body, method=self.command,
            headers={k: v for k, v in self.headers.items()
                     if k.lower() not in ("host", "transfer-encoding")},
        )
        req.add_header("Host", f"localhost:{LISTEN_PORT}")

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ("transfer-encoding",):
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(resp.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for k, v in e.headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_error(502, str(e))

    do_GET = _proxy
    do_POST = _proxy
    do_PUT = _proxy
    do_DELETE = _proxy

    def log_message(self, format, *args):
        pass  # quiet


class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    server = ThreadedServer(("0.0.0.0", LISTEN_PORT), LBHandler)
    print(f"Load balancer on :{LISTEN_PORT} -> {BACKENDS}", flush=True)
    server.serve_forever()
