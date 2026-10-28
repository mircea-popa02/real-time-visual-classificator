"""The daemon: wires sources, processors, presence, rules, watches and the store together.

Each loop is a `*_step` method driven by `run()`; tests call the steps directly with a fake clock.
"""

import asyncio
from enum import Enum
import logging
import threading
import time

from .decision import DecisionError
from .events import Event, EventBus
from .model import INTERACTIVE, PRESENCE, WATCH
from .presence import PresenceFSM, PresenceState, idle_probability, idle_seconds
from .processors.screen_text import ScreenTextProcessor
from .processors.speech import Segmenter, SpeechProcessor, silero_vad
from .processors.summary import Throttle, frame_uri, scene_note, summarize
from .rules import RulesEngine, run_detached
from .sources.screen import SourceError
from .watches import WatchRegistry, source_matches

log = logging.getLogger(__name__)

_FAILURES_TO_OFF = 3
_RETRY_AFTER = 60.0
_WATCH_EVERY = 2.0
_STATE = "Judge only what is visibly present. If uncertain, favor absent."


class CaptureState(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    PAUSED_ALL = "paused_all"


class _Source:
    def __init__(self, backend, label=None):
        self.backend = (label or getattr(backend, "name", type(backend).__name__)) if backend is not None else None
        self.state = "on" if backend is not None else "off"
        self.error = None if backend is not None else "not available"
        self.failures = 0
        self.retry_at = None

    def as_dict(self):
        return {"backend": self.backend, "state": self.state, "error": self.error}


class Daemon:
    def __init__(self, config, store, client, screen, camera, audio, ocr, clock=time.time, sleep=asyncio.sleep,
                 run=run_detached, which=None):
        self.config = config
        self.store = store
        self.client = client
        self.screen = screen
        self.camera = camera
        self.audio = [stream for stream in audio if stream is not None]
        self.clock = clock
        self.sleep = sleep
        self.bus = EventBus()
        self.watches = WatchRegistry(clock=clock)
        self.fsm = PresenceFSM(config.presence)
        rules_kwargs = {"run": run} if which is None else {"run": run, "which": which}
        self.rules = RulesEngine(config.rules, **rules_kwargs)
        self.screen_text = ScreenTextProcessor(ocr, config.screen.min_change, config.screen.ocr_per_minute)
        self.capture = CaptureState.ACTIVE
        self.paused_until = None
        self.sources = {"screen": _Source(screen), "camera": _Source(camera)}
        for name in ("mic", "desktop"):
            stream = next((s for s in self.audio if s.name == name), None)
            self.sources[name] = _Source(stream, getattr(getattr(stream, "device", None), "name", None))
        self._summary = Throttle(config.summary.min_interval)
        self._scene = Throttle(config.summary.scene_interval)
        self._screen_watch = Throttle(_WATCH_EVERY)
        self._camera_watch = Throttle(_WATCH_EVERY)
        self._lock_held = False
        self._last_purge = None
        self._stopping = False
        self._loop = None
        self._loop_thread = None

    # -- events ---------------------------------------------------------------------------------------------

    def emit(self, event):
        if self._loop is not None and threading.get_ident() != self._loop_thread:
            self._loop.call_soon_threadsafe(self._handle, event)
        else:
            self._handle(event)

    def _handle(self, event):
        if event.meta.get("store") is not False:
            self.store.add(event)
        self.bus.publish(event)
        now = self.clock()
        for watch in self.watches.on_event(event):
            self._handle(Event(now, "system", "watch.fired",
                               f"{watch.id} ({watch.condition.summary()}): {event.text or event.kind}",
                               {"watch": watch.id}))
        self.rules.on_event(event, now)

    # -- state ----------------------------------------------------------------------------------------------

    @property
    def present(self):
        return self.fsm.state == PresenceState.PRESENT

    def pause(self, duration=None, all=False):
        self.capture = CaptureState.PAUSED_ALL if all else CaptureState.PAUSED
        self.paused_until = self.clock() + duration if duration else None
        self.rules.suspended = all
        self.emit(Event(self.clock(), "system", "capture.paused", "everything" if all else "capture",
                        {"all": all, "until": self.paused_until}))

    def resume(self):
        self.capture = CaptureState.ACTIVE
        self.paused_until = None
        self.rules.suspended = False
        self.emit(Event(self.clock(), "system", "capture.resumed"))

    def _capturing(self):
        return self.present and self.capture == CaptureState.ACTIVE

    # -- sources --------------------------------------------------------------------------------------------

    async def _use(self, name, fn):
        """Run a blocking source call; three failures in a row turn the source off, re-probed every 60 s."""
        source = self.sources[name]
        now = self.clock()
        if source.backend is None or (source.state == "off" and source.retry_at and now < source.retry_at):
            return None
        try:
            result = await asyncio.to_thread(fn)
        except SourceError as exc:
            source.failures += 1
            source.error = str(exc)
            if source.state == "off":
                source.retry_at = now + _RETRY_AFTER
            elif source.failures >= _FAILURES_TO_OFF:
                source.state, source.retry_at = "off", now + _RETRY_AFTER
                self.emit(Event(now, "system", "source.off", f"{name}: {exc}", {"source": name}))
            return None
        if source.state == "off":
            source.state, source.error, source.retry_at = "on", None, None
            self.emit(Event(now, "system", "source.on", name, {"source": name}))
        source.failures = 0
        return result

    async def _ask(self, frame, questions, priority):
        try:
            return await asyncio.wrap_future(self.client.ask([frame_uri(frame)], questions, _STATE, priority))
        except (DecisionError, OSError, ValueError) as exc:
            log.warning("Model question failed: %s", exc)
            return None

    async def _question_watches(self, frame, source_name):
        watches = [w for w in self.watches.question_watches() if source_matches(source_name, w.condition.sources)]
        if not watches:
            return
        questions = {f"watch:{w.id}": {"type": "binary", "instructions": w.condition.question} for w in watches}
        result = await self._ask(frame, questions, WATCH)
        if not result:
            return
        event = Event(frame.ts or self.clock(), source_name, "watch.seen")
        for watch in watches:
            probability = result["answers"][f"watch:{watch.id}"]["probability"]
            if self.watches.report_score(watch.id, probability, event):
                self.emit(Event(self.clock(), "system", "watch.fired",
                                f"{watch.id} ({watch.condition.summary()}): {probability:.0%}", {"watch": watch.id}))

    # -- loops ----------------------------------------------------------------------------------------------

    async def presence_step(self):
        if self.capture == CaptureState.PAUSED_ALL:
            return
        now = self.clock()
        probability = None
        if self.camera is not None and self.sources["camera"].backend is not None:
            frame = await self._use("camera", self.camera.read)
            if frame is not None:
                result = await self._ask(frame, {"present": {"type": "binary",
                                                             "instructions": self.config.presence.question}},
                                         PRESENCE)
                probability = result["answers"]["present"]["probability"] if result else None
        else:
            probability = idle_probability(await asyncio.to_thread(idle_seconds))
        transition = self.fsm.update(probability, now)
        if transition is not None:
            self.emit(Event(now, "presence", f"presence.{transition.value}"))
            if transition == PresenceState.AWAY:
                self.watches.cap_ttl(1800, now)

    async def screen_step(self):
        away_watch = (not self.present and self.capture == CaptureState.ACTIVE and self.watches.has_screen_watch())
        if self.screen is None or not (self._capturing() or away_watch):
            return
        frames = await self._use("screen", self.screen.capture)
        for frame in frames or []:
            frame.ts = frame.ts or self.clock()
            event = await asyncio.to_thread(self.screen_text.process, frame)
            if event is not None:
                if not self.present:
                    event = Event(event.ts, event.source, event.kind, event.text, {**event.meta, "store": False})
                self.emit(event)
                if self._capturing() and self._summary.ready(self.clock()):
                    summarize(self.client, frame).add_done_callback(self._emit_future)
            if self._screen_watch.ready(self.clock()):
                await self._question_watches(frame, f"screen:{frame.name}")

    async def camera_step(self):
        if self.camera is None or self.capture == CaptureState.PAUSED_ALL:
            return
        scene = self._capturing() and self._scene.ready(self.clock())
        watching = any(w.condition.sources == ["camera"] for w in self.watches.question_watches())
        if not (scene or watching):
            return
        frame = await self._use("camera", self.camera.read)
        if frame is None:
            return
        frame.ts = frame.ts or self.clock()
        if scene:
            scene_note(self.client, frame).add_done_callback(self._emit_future)
        if watching and self._camera_watch.ready(self.clock()):
            await self._question_watches(frame, "camera")

    def _emit_future(self, future):
        try:
            self.emit(future.result())
        except Exception as exc:
            log.warning("Background model call failed: %s", exc)

    def housekeeping(self):
        now = self.clock()
        if self.paused_until is not None and now >= self.paused_until:
            self.resume()
        hold = not self.present and self.watches.has_screen_watch() and self.capture != CaptureState.PAUSED_ALL
        events = []
        if hold and not self._lock_held:
            self.rules.hold("@screen.lock")
        elif not hold and self._lock_held:
            events += self.rules.release("@screen.lock", now)
        self._lock_held = hold
        events += self.rules.tick(now)
        for event in events:
            self.emit(event)
        self.watches.expire(now)
        if self._last_purge is None or now - self._last_purge >= 3600:
            self._last_purge = now
            self.store.purge(now - self.config.store.retention_days * 86400)

    def _audio_worker(self, stream):
        try:
            segmenter = Segmenter(silero_vad(), max_length=self.config.audio.max_utterance)
            speech = SpeechProcessor(self.client, self.emit, stream.name)
            for block in stream.blocks():
                if self._stopping:
                    return
                if not self._capturing():
                    continue
                for utterance in segmenter.feed(block):
                    speech.handle(utterance, self.clock())
        except Exception as exc:
            log.warning("Audio source %s stopped: %s", stream.name, exc)
            self.sources[stream.name].state, self.sources[stream.name].error = "off", str(exc)
            self.emit(Event(self.clock(), "system", "source.off", f"{stream.name}: {exc}", {"source": stream.name}))

    async def _loop_forever(self, step, interval):
        while not self._stopping:
            try:
                result = step()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                log.exception("%s failed", getattr(step, "__name__", step))
            await self.sleep(interval())

    async def run(self):
        self._loop = asyncio.get_running_loop()
        self._loop_thread = threading.get_ident()
        for stream in self.audio:
            threading.Thread(target=self._audio_worker, args=(stream,), daemon=True, name=f"audio-{stream.name}").start()
        await asyncio.gather(
            self._loop_forever(self.presence_step, lambda: self.fsm.interval(self.clock())),
            self._loop_forever(self.screen_step, lambda: self.config.screen.interval),
            self._loop_forever(self.camera_step, lambda: _WATCH_EVERY),
            self._loop_forever(self.housekeeping, lambda: 0.5),
        )

    def stop(self):
        self._stopping = True
        for stream in self.audio:
            stream.close()

    # -- queries --------------------------------------------------------------------------------------------

    def status(self):
        return {
            "presence": self.fsm.state.value,
            "capture": self.capture.value,
            "paused_until": self.paused_until,
            "sources": {name: source.as_dict() for name, source in self.sources.items()},
            "model": {"url": getattr(self.client, "base_url", None), "healthy": self.client.healthy()},
            "watches": len(self.watches.list()),
            "lock_held_by_watch": self._lock_held or (not self.present and self.watches.has_screen_watch()),
            "exclusions": "none",
        }

    def now(self, sources=None):
        wanted = sources or ["screen", "mic", "desktop", "camera"]
        screens = {}
        for name in self.screen_text.screens():
            if source_matches(f"screen:{name}", wanted):
                summary = self.store.latest("screen.summary", source=f"screen:{name}")
                screens[name] = {"text": "\n".join(self.screen_text.lines(name)),
                                 "summary": summary.text if summary else None}
        now = self.clock()
        speech = [e for e in self.store.timeline(now - 300, sources=[s for s in ("mic", "desktop") if s in wanted])
                  if e.kind.startswith("speech.")][-10:]
        scene = self.store.latest("camera.scene") if "camera" in wanted else None
        return {"presence": self.fsm.state.value, "screens": screens, "recent_speech": speech,
                "scene": scene.text if scene else None}

    async def _fresh(self, source):
        if source == "camera":
            if self.camera is None:
                raise ValueError("no camera")
            return await asyncio.to_thread(self.camera.read)
        if not source_matches(source, ["screen"]):
            raise ValueError(f"unknown source {source!r}; use camera, screen or screen:<name>")
        if self.screen is None:
            raise ValueError("no screen capture available")
        frames = await asyncio.to_thread(self.screen.capture)
        name = source.partition(":")[2]
        chosen = [f for f in frames if f.name == name] if name else frames
        if not chosen:
            raise ValueError(f"no screen named {name!r}; have {', '.join(f.name for f in frames)}")
        frame = chosen[0]
        frame.ts = frame.ts or self.clock()
        return frame

    async def look(self, source, prompt=None):
        frame = await self._fresh(source)
        if prompt:
            return await asyncio.wrap_future(
                self.client.describe([frame_uri(frame)], prompt, INTERACTIVE, max_tokens=160))
        if source == "camera":
            return (await asyncio.wrap_future(scene_note(self.client, frame))).text
        await asyncio.to_thread(self.screen_text.process, frame)
        summary = await asyncio.wrap_future(summarize(self.client, frame))
        return f"{summary.text}\n\n" + "\n".join(self.screen_text.lines(frame.name))

    async def ask(self, source, questions):
        frame = await self._fresh(source)
        return await asyncio.wrap_future(self.client.ask([frame_uri(frame)], questions, _STATE, INTERACTIVE))
