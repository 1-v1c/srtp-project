"""Loopback-only browser UI transport for the existing DMS pipeline."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Condition, Event, Thread
from time import monotonic
from urllib.parse import urlsplit
import webbrowser

import cv2


WEB_ROOT = Path(__file__).resolve().parents[1] / "dashboard"
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/dashboard.css": ("dashboard.css", "text/css; charset=utf-8"),
    "/dashboard.js": ("dashboard.js", "text/javascript; charset=utf-8"),
}


class Dashboard:
    def __init__(self):
        self.stop = Event()
        self.paused = Event()
        self._updates = Condition()
        self._actions = Queue(maxsize=12)
        self._state = {"status": "starting", "message": "正在初始化摄像头和模型"}
        self._jpeg = None
        self._version = 0
        self._next_frame_at = 0.0

    def status(self, status, message):
        with self._updates:
            self._state = {"status": status, "message": message}
            self._jpeg = None
            self._updates.notify_all()

    def publish(self, frame, telemetry):
        now = monotonic()
        if now < self._next_frame_at:
            return
        self._next_frame_at = now + 1 / 18
        ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        with self._updates:
            self._state = {"status": "live", **telemetry}
            self._jpeg = data.tobytes()
            self._version += 1
            self._updates.notify_all()

    def snapshot(self):
        with self._updates:
            return self._state, self._jpeg

    def next_frame(self, version):
        with self._updates:
            self._updates.wait_for(lambda: self._version != version or self.stop.is_set()
                                   or self._state["status"] != "live", timeout=1)
            return self._version, self._jpeg if version != self._version else None

    def queue_action(self, action, value):
        if action not in ("pause", "resume", "stop"):
            self._actions.put_nowait((action, value))
            return
        with self._updates:
            if action == "pause":
                self.paused.set()
            elif action == "resume":
                self.paused.clear()
            else:
                self.stop.set()
            self._updates.notify_all()

    def wait_for_resume(self):
        with self._updates:
            self._updates.wait_for(lambda: not self.paused.is_set() or self.stop.is_set())
            return not self.stop.is_set()

    def pending_actions(self):
        while True:
            try:
                yield self._actions.get_nowait()
            except Empty:
                return


def create_server(dashboard, port):
    class Handler(BaseHTTPRequestHandler):
        def allowed_host(self):
            addresses = {f"127.0.0.1:{self.server.server_port}",
                         f"localhost:{self.server.server_port}"}
            return self.headers.get("Host") in addresses

        def respond(self, status, content_type, content, **headers):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            for name, value in headers.items():
                self.send_header(name.replace("_", "-"), value)
            self.end_headers()
            self.wfile.write(content)

        def json_response(self, status, body):
            self.respond(status, "application/json; charset=utf-8",
                         json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8"))

        def do_GET(self):
            if not self.allowed_host():
                self.send_error(403)
                return
            path = urlsplit(self.path).path
            try:
                self.route_get(path)
            except (ConnectionError, TimeoutError):
                pass  # A tab can close while a response is being streamed.

        def route_get(self, path):
            if path in ASSETS:
                file, content_type = ASSETS[path]
                self.respond(200, content_type, (WEB_ROOT / file).read_bytes(),
                             Content_Security_Policy="default-src 'self'; object-src 'none'; frame-ancestors 'none'")
                return
            state, jpeg = dashboard.snapshot()
            if path == "/api/state":
                self.json_response(200, state)
            elif path == "/api/frame.jpg" and jpeg is not None:
                self.respond(200, "image/jpeg", jpeg,
                             Content_Disposition='attachment; filename="dms-snapshot.jpg"')
            elif path == "/api/stream" and jpeg is not None:
                self.stream_frames()
            elif path in ("/api/stream", "/api/frame.jpg"):
                self.send_error(503, "Camera has no frame yet")
            else:
                self.send_error(404)

        def stream_frames(self):
            self.connection.settimeout(3)
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            version = -1
            while not dashboard.stop.is_set():
                version, jpeg = dashboard.next_frame(version)
                if jpeg is not None:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                                     + jpeg + b"\r\n")
                if dashboard.snapshot()[0]["status"] != "live":
                    break

        def do_POST(self):
            origin = self.headers.get("Origin")
            if (not self.allowed_host() or self.headers.get("X-DMS-Action") != "1"
                    or (origin and origin != f"http://{self.headers.get('Host')}")):
                self.send_error(403)
                return
            if self.path != "/api/actions":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 256:
                    raise ValueError
                body = json.loads(self.rfile.read(size))
                action, value = body["action"], body.get("enabled")
                if action not in ("reset", "calibrate", "mesh", "distraction", "pause", "resume", "stop"):
                    raise ValueError
                if action in ("mesh", "distraction") and not isinstance(value, bool):
                    raise ValueError
                allowed_status = ("paused",) if action == "resume" else ("live",)
                if action == "stop":
                    allowed_status = ("live", "paused", "resuming")
                if dashboard.snapshot()[0]["status"] not in allowed_status:
                    self.json_response(409, {"error": "当前监测状态无法执行该操作"})
                    return
                dashboard.queue_action(action, value)
            except (ValueError, KeyError, TypeError, AttributeError):
                self.json_response(400, {"error": "无效的操作"})
                return
            except Full:
                self.json_response(409, {"error": "操作过于频繁"})
                return
            self.json_response(202, {"accepted": True})

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def run_dashboard(run_capture, port, camera_id, open_browser):
    dashboard = Dashboard()

    def capture():
        try:
            run_capture(dashboard=dashboard, camera_id=camera_id)
        except Exception as error:
            dashboard.status("error", str(error))
            print(f"[ERROR] {error}", flush=True)

    with create_server(dashboard, port) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"[DMS] {url}  ·  Ctrl+C 退出", flush=True)
        worker = Thread(target=capture, name="dms-capture", daemon=True)
        worker.start()
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever(poll_interval=0.25)
        except KeyboardInterrupt:
            pass
        finally:
            dashboard.queue_action("stop", None)
            worker.join(timeout=5)
