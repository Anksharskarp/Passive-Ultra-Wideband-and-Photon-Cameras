#!/usr/bin/env python3
"""Local browser interface. Run: python interactive.py, then open the printed URL."""

import argparse
from collections import OrderedDict
import io
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from urllib.parse import urlparse, parse_qs
import uuid

import numpy as np

from experiment import create_experiment, spectrum, summary, view
from photon_simulator import save_events


STATIC = Path(__file__).resolve().parent / "web"
RUNS = OrderedDict()
LOCK = threading.Lock()
MAX_REQUEST = 4_000_000
FILES = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
         "/style.css": ("style.css", "text/css")}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        if len(args) < 2 or str(args[1]) not in {"200", "204"}:
            super().log_message(format, *args)

    def reply(self, status, data, content_type="application/json", attachment=None):
        if content_type == "application/json":
            data = json.dumps(data, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def same_host(self):
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        origin = self.headers.get("Origin")
        return self.headers.get("Host") in allowed and (
            not origin or urlparse(origin).netloc in allowed)

    def do_GET(self):
        if not self.same_host():
            return self.reply(403, {"error": "Use the local application address."})
        parsed = urlparse(self.path)
        if parsed.path in FILES:
            filename, mime = FILES[parsed.path]
            return self.reply(200, (STATIC/filename).read_bytes(), mime)
        if parsed.path == "/favicon.ico":
            return self.reply(204, b"", "image/x-icon")
        if parsed.path == "/api/export":
            with LOCK:
                run = RUNS.get(parse_qs(parsed.query).get("id", [""])[0])
            if run is None:
                return self.reply(404, {"error": "This run expired; simulate again."})
            # Include full input/config so media runs can also be reproduced.
            stream = io.BytesIO()
            save_events(stream, run.result)
            stream.seek(0)
            with np.load(stream, allow_pickle=False) as archive:
                values = {name: archive[name] for name in archive.files}
            values["config_json"] = np.array(json.dumps(run.config))
            output = io.BytesIO()
            np.savez_compressed(output, **values)
            return self.reply(200, output.getvalue(), "application/octet-stream", "photons.npz")
        self.reply(404, {"error": "Not found."})

    def do_POST(self):
        if not self.same_host():
            return self.reply(403, {"error": "Use the local application address."})
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self.reply(415, {"error": "Expected JSON."})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_REQUEST:
                raise ValueError("Request is empty or larger than 4 MB.")
            config = json.loads(self.rfile.read(size))
            if not isinstance(config, dict):
                raise ValueError("Expected a JSON object.")
            with LOCK:
                if self.path == "/api/run":
                    run = create_experiment(config)
                    initial_view = config.get("view", {})
                    data = view(run, initial_view)
                    spec = spectrum(run, *data["pixel"])
                    identifier = uuid.uuid4().hex
                    RUNS[identifier] = run
                    while len(RUNS) > 3:
                        RUNS.popitem(last=False)
                    response = {"id": identifier, "summary": summary(run), "view": data,
                                "spectrum": spec, "config": run.config}
                elif self.path == "/api/view":
                    run = RUNS.get(config.get("id"))
                    if run is None:
                        return self.reply(404, {"error": "This run expired; simulate again."})
                    response = {"view": view(run, config)}
                    if config.get("spectrum"):
                        response["spectrum"] = spectrum(run, *response["view"]["pixel"])
                else:
                    return self.reply(404, {"error": "Not found."})
            self.reply(200, response)
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            self.reply(400, {"error": str(error)})
        except Exception:
            import traceback
            traceback.print_exc()
            self.reply(500, {"error": "Simulation failed. See the terminal for details."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as error:
        parser.exit(1, f"Cannot start server: {error}. Try --port 8766.\n")
    print(f"Photon arrival lab → http://127.0.0.1:{server.server_port}", flush=True)
    print("Ctrl+C stops the server. All simulations and uploads stay on this computer.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
