import http.client
import json
from pathlib import Path
import sys
from threading import Thread
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from web_dashboard import Dashboard, create_server


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.dashboard = Dashboard()
        self.server = create_server(self.dashboard, 0)
        self.thread = Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.dashboard.queue_action("stop", None)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        try:
            conn.request(method, path, body, headers or {})
            response = conn.getresponse()
            return response.status, response.getheader("Content-Type"), response.read()
        finally:
            conn.close()

    def publish(self):
        self.dashboard.publish(np.zeros((24, 32, 3), dtype=np.uint8), {"level": 2, "ear": None})

    def test_local_assets_and_telemetry(self):
        for path in ("/", "/dashboard.css", "/dashboard.js"):
            self.assertEqual(self.request("GET", path)[0], 200)
        self.assertEqual(self.request("GET", "/api/frame.jpg")[0], 503)
        self.publish()
        status, _, data = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data), {"status": "live", "level": 2, "ear": None})

    def test_stream_and_snapshot_return_jpeg(self):
        self.publish()
        status, content_type, payload = self.request("GET", "/api/frame.jpg")
        self.assertEqual((status, content_type), (200, "image/jpeg"))
        self.assertTrue(payload.startswith(b"\xff\xd8"))
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        try:
            conn.request("GET", "/api/stream")
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            self.assertIn("multipart/x-mixed-replace", response.getheader("Content-Type"))
            self.assertEqual(response.read(7), b"--frame")
        finally:
            conn.close()

    def test_controls_and_external_origin_rejection(self):
        self.publish()
        self.assertEqual(self.request("GET", "/", headers={"Host": "example.test"})[0], 403)
        headers = {"X-DMS-Action": "1", "Content-Type": "application/json"}
        rejected = {**headers, "Origin": "https://example.test"}
        path = "/api/actions"
        self.assertEqual(self.request("POST", path, '{"action":"reset"}', rejected)[0], 403)
        self.assertEqual(self.request("POST", path, '{"action":"mesh","enabled":"off"}', headers)[0], 400)
        self.assertEqual(self.request("POST", path, '{"action":"mesh","enabled":false}', headers)[0], 202)
        self.assertEqual(list(self.dashboard.pending_actions()), [("mesh", False)])
        self.assertEqual(self.request("POST", path, '{"action":"stop"}', headers)[0], 202)
        self.assertTrue(self.dashboard.stop.is_set())


if __name__ == "__main__":
    unittest.main()
