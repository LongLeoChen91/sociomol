#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launch SocioMol Arm Builder using Python's built-in HTTP server.

No external dependencies required — uses only the standard library.

Usage:
    python serve.py
    python serve.py --port 8080
"""

import argparse
import functools
import http.server
import pathlib
import webbrowser

STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static"


def main():
    parser = argparse.ArgumentParser(
        description="Serve the SocioMol Arm Builder web app locally.",
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Port to serve on (default: 5000).",
    )
    parser.add_argument(
        "--no-open", action="store_true",
        help="Do not automatically open the browser.",
    )
    args = parser.parse_args()

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(STATIC_DIR),
    )

    url = f"http://127.0.0.1:{args.port}"
    print(f"SocioMol Arm Builder serving at {url}")
    print("Press Ctrl+C to stop.\n")

    if not args.no_open:
        webbrowser.open(url)

    server = http.server.HTTPServer(("127.0.0.1", args.port), handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
