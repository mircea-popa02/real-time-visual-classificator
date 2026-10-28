import sys
from pathlib import Path
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from starlette.testclient import TestClient

from fakes import FakeCamera, FakeClient, FakeOcr, FakeScreen
from senses.config import Config
from senses.daemon import Daemon
from senses.events import Event
from senses.mcp_server import build_server
from senses.store import Store


def text_of(result):
    return result.content[0].text


class McpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.daemon = Daemon(Config(), Store(":memory:"), FakeClient(), FakeScreen(), FakeCamera(), (None, None),
                             FakeOcr({0: "hello"}), run=lambda argv: None)
        self.server = build_server(self.daemon)
        self.daemon.store.add(Event(time.time() - 60, "screen:desktop", "screen.text", "Deployment complete"))

    async def call(self, name, args=None):
        return text_of(await self.server.call_tool(name, args or {}))

    async def test_search_formats_events(self):
        out = await self.call("search", {"query": "deployment"})
        self.assertIn("screen:desktop", out)
        self.assertIn("Deployment complete", out)

    async def test_timeline_default_window(self):
        self.assertIn("Deployment complete", await self.call("timeline"))

    async def test_bad_since_is_an_error_line(self):
        self.assertTrue((await self.call("timeline", {"since": "whenever"})).startswith("error:"))

    async def test_watch_then_wait_still_waiting(self):
        out = await self.call("watch", {"condition": {"text": "error"}})
        wid = out.split()[1]
        self.assertEqual(await self.call("wait_for", {"watch_id": wid, "timeout": 1}), f"still waiting (watch {wid})")
        self.assertIn(wid, await self.call("watches"))
        self.assertEqual(await self.call("unwatch", {"watch_id": wid}), "removed")
        self.assertTrue((await self.call("unwatch", {"watch_id": wid})).startswith("error:"))

    async def test_wait_for_fired(self):
        wid = (await self.call("watch", {"condition": {"text": "build ok"}})).split()[1]
        self.daemon.emit(Event(time.time(), "screen:desktop", "screen.text", "build ok"))
        out = await self.call("wait_for", {"watch_id": wid, "timeout": 1})
        self.assertTrue(out.startswith("fired:"))
        self.assertIn("build ok", out)

    async def test_wait_for_timeout_clamped(self):
        wid = (await self.call("watch", {"condition": {"text": "x"}})).split()[1]
        seen = []

        async def fake_wait(id, timeout):
            seen.append(timeout)
            return None

        with patch.object(self.daemon.watches, "wait", fake_wait):
            await self.call("wait_for", {"watch_id": wid, "timeout": 500})
            await self.call("wait_for", {"watch_id": wid, "timeout": 0})
        self.assertEqual(seen, [90, 1])

    async def test_invalid_condition_and_unknown_watch(self):
        self.assertTrue((await self.call("watch", {"condition": {"color": "red"}})).startswith("error:"))
        self.assertTrue((await self.call("wait_for", {"watch_id": "nope"})).startswith("error:"))

    async def test_status_pause_resume(self):
        self.assertIn("capture: active", await self.call("status"))
        self.assertIn("paused", await self.call("pause", {"duration": "10m"}))
        self.assertEqual(self.daemon.status()["capture"], "paused")
        await self.call("resume")
        self.assertEqual(self.daemon.status()["capture"], "active")

    async def test_ask_formats_probabilities(self):
        out = await self.call("ask", {"source": "camera",
                                      "questions": {"cup": {"type": "binary", "instructions": "Is there a cup?"}}})
        self.assertEqual(out, "cup: 90% yes")

    async def test_now_lists_presence_and_screens(self):
        out = await self.call("now")
        self.assertIn("presence: away", out)

    async def test_look_unknown_source(self):
        self.assertTrue((await self.call("look", {"source": "fridge"})).startswith("error:"))


class ControlRouteTests(unittest.TestCase):
    def setUp(self):
        self.daemon = Daemon(Config(), Store(":memory:"), FakeClient(), FakeScreen(), None, (None, None),
                             FakeOcr(), run=lambda argv: None)
        self.daemon.store.add(Event(time.time() - 5, "mic", "speech.me", "salut lume"))
        self.http = TestClient(build_server(self.daemon).streamable_http_app())

    def test_status_and_pause(self):
        self.assertEqual(self.http.get("/control/status").json()["capture"], "active")
        self.assertEqual(self.http.post("/control/pause", json={"all": True}).json()["capture"], "paused_all")
        self.assertEqual(self.http.post("/control/resume").json()["capture"], "active")

    def test_search_and_timeline(self):
        self.assertEqual(self.http.get("/control/search", params={"q": "salut"}).json()[0]["text"], "salut lume")
        self.assertEqual(self.http.get("/control/timeline", params={"since": "1m"}).json()[0]["source"], "mic")
        self.assertEqual(self.http.get("/control/timeline", params={"since": "soon"}).status_code, 400)

    def test_history_page_is_served(self):
        page = self.http.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("senses history", page.text)
        self.assertIn("/control/timeline", page.text)

    def test_timeline_source_filter_and_limit(self):
        self.daemon.store.add(Event(time.time() - 4, "screen:desktop", "screen.text", "editor open"))
        self.daemon.store.add(Event(time.time() - 3, "screen:desktop", "screen.text", "terminal open"))
        rows = self.http.get("/control/timeline", params={"since": "1m", "sources": "screen", "limit": 1}).json()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "screen:desktop")
        self.assertEqual(rows[0]["kind"], "screen.text")

    def test_search_respects_sources(self):
        self.daemon.store.add(Event(time.time() - 4, "screen:desktop", "screen.text", "salut screen"))
        rows = self.http.get("/control/search", params={"q": "salut", "sources": "mic"}).json()
        self.assertEqual([row["text"] for row in rows], ["salut lume"])


if __name__ == "__main__":
    unittest.main()
