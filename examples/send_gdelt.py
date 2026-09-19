"""Send a CSV to Fuego using only the Python standard library."""

import argparse
import json
import os
from pathlib import Path
import socket
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def send_csv(path, host="127.0.0.1", port=8000, transport="http", token="", timeout=900):
    raw = Path(path).read_bytes()
    if transport == "tcp":
        frame = {"format": "csv", "data": raw.decode("utf-8-sig"), "token": token}
        with socket.create_connection((host, port), timeout=timeout) as conn:
            conn.sendall(json.dumps(frame).encode() + b"\n")
            with conn.makefile("rb") as response:
                return json.loads(response.readline())
    request = Request(f"http://{host}:{port}/v1/gdelt", data=raw,
                      headers={"Content-Type": "text/csv; charset=utf-8", "Authorization": "Bearer " + token})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        with exc:
            return json.load(exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--transport", choices=("http", "tcp"), default="http")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    try:
        result = send_csv(args.path, args.host, args.port, args.transport,
                          os.environ.get("FUEGO_API_TOKEN", ""), args.timeout)
    except (OSError, URLError, ValueError) as exc:
        print(f"Send failed: {type(exc).__name__}. Check the address and server status.", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
