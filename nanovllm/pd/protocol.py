"""Framed JSON socket protocol shared by the PD serving roles and harnesses.

Frame = [u32 length][JSON payload].  ``JsonStream`` parses incrementally so
either side can interleave socket reads with engine steps.
"""
import json
import socket
import struct
import time

HDR = struct.Struct("<I")


def send_json(sock, obj):
    data = json.dumps(obj).encode()
    sock.sendall(HDR.pack(len(data)) + data)


class JsonStream:
    """Incremental [u32][json] frame parser; owns its own byte buffer."""

    def __init__(self):
        self._buf = bytearray()
        self._state = "hdr"
        self._need = HDR.size

    def feed(self, chunk):
        self._buf += chunk
        out = []
        while len(self._buf) >= self._need:
            if self._state == "hdr":
                (n,) = HDR.unpack(self._buf[:HDR.size])
                del self._buf[:HDR.size]
                self._need = n
                self._state = "payload"
            else:
                payload = bytes(self._buf[:self._need])
                del self._buf[:self._need]
                out.append(json.loads(payload))
                self._need = HDR.size
                self._state = "hdr"
        return out


def connect_retry(host, port, tries=120):
    for i in range(tries):
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.setblocking(False)
            return sock
        except OSError:
            time.sleep(1.0 if i < 15 else 2.0)
    raise ConnectionError(f"cannot connect {host}:{port}")


def recvn_blocking(sock, n):
    chunks = []
    while n > 0:
        chunk = sock.recv(min(n, 1 << 22))
        if not chunk:
            raise ConnectionError("peer closed")
        chunks.append(chunk)
        n -= len(chunk)
    return b"".join(chunks)
