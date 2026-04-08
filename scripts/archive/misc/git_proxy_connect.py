#!/usr/bin/env python3
"""
Git GIT_PROXY_COMMAND helper: establishes a CONNECT tunnel through an HTTP proxy.
Git calls this as: git_proxy_connect.py <host> <port>
It must connect stdin/stdout to the tunneled connection.
"""
import base64, os, socket, sys

PROXY_HOST = "192.168.189.225"
PROXY_PORT = 1080
PROXY_USER = "user"
PROXY_PASS = "passWorD"

def main():
    host = sys.argv[1]
    port = int(sys.argv[2])

    sock = socket.create_connection((PROXY_HOST, PROXY_PORT), timeout=10)

    auth = base64.b64encode(f"{PROXY_USER}:{PROXY_PASS}".encode()).decode()
    connect_req = (
        f"CONNECT {host}:{port} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Proxy-Authorization: Basic {auth}\r\n"
        f"User-Agent: curl/7.88.1\r\n"
        f"\r\n"
    )
    sock.sendall(connect_req.encode())

    response = b""
    while b"\r\n\r\n" not in response:
        chunk = sock.recv(4096)
        if not chunk:
            sys.exit(1)
        response += chunk

    status_line = response.split(b"\r\n")[0].decode()
    if "200" not in status_line:
        sys.stderr.write(f"Proxy CONNECT failed: {status_line}\n")
        sys.exit(1)

    # Relay stdin/stdout <-> socket
    import select, threading

    def relay(src, dst_write):
        try:
            while True:
                data = src.read(65536) if hasattr(src, 'read') else os.read(src, 65536)
                if not data:
                    break
                if isinstance(dst_write, socket.socket):
                    dst_write.sendall(data)
                else:
                    os.write(dst_write, data)
        except:
            pass

    def sock_to_stdout():
        try:
            while True:
                data = sock.recv(65536)
                if not data:
                    break
                os.write(1, data)
        except:
            pass

    t = threading.Thread(target=sock_to_stdout, daemon=True)
    t.start()

    try:
        while True:
            data = os.read(0, 65536)
            if not data:
                break
            sock.sendall(data)
    except:
        pass

if __name__ == "__main__":
    main()
