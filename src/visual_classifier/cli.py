"""Classify webcam frames or images with a local vision model."""

import argparse
import base64
import concurrent.futures
import datetime as dt
import json
import mimetypes
import os
from pathlib import Path
import queue
import sys
import threading
import time

from .catalog import OBJECTS, SCENE_QUESTIONS, object_questions
from .decision import DecisionError, local_decide, local_observe


def image_uri(path):
    path = Path(path).expanduser()
    mime, _ = mimetypes.guess_type(path)
    if mime not in {"image/jpeg", "image/png", "image/webp"}:
        raise DecisionError("Image must be JPEG, PNG, or WebP")
    data = path.read_bytes()
    if len(data) > 8 * 1024 * 1024:
        raise DecisionError("Image exceeds 8 MiB; resize it before scoring")
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def render(result, threshold):
    answers = result["answers"]
    present = sorted(((name, answer["probability"]) for name, answer in answers.items()
                      if answer.get("type") == "binary" and answer.get("probability", 0) >= threshold),
                     key=lambda pair: pair[1], reverse=True)
    scene = {name: answer.get("choice", answer.get("score")) for name, answer in answers.items()
             if answer.get("type") in {"choice", "score"}}
    return {"objects": [{"name": name, "probability": round(value, 4)} for name, value in present],
            "scene": scene, "answers": answers}


def _questions(args):
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not isinstance(config.get("questions"), dict):
            raise DecisionError("Config must contain a questions object")
        return config["questions"], config.get("state", "Judge only what is visibly present.")
    objects = OBJECTS if args.objects == "all" else tuple(x.strip() for x in args.objects.split(","))
    questions = object_questions(objects)
    if args.scene:
        questions.update(SCENE_QUESTIONS)
    return questions, "Inspect the image. Judge only what is visibly present. If uncertain, favor absent."


def _decide(args, state, questions, images, progress=None):
    started = time.perf_counter()
    model = args.model or "local-vision"
    names = None
    observation = None
    if images and not args.config and args.objects == "all":
        observation = local_observe(images, OBJECTS, args.url, model,
                                    os.environ.get("LOCAL_API_KEY"), args.timeout,
                                    limit=args.max_candidates)
        names = observation["objects"]
        if progress:
            progress("observation", observation)
        questions = object_questions(names) if names else {}
        if args.scene:
            questions.update(SCENE_QUESTIONS)
    result = (local_decide(state, questions, images, args.url, model,
                           os.environ.get("LOCAL_API_KEY"), args.timeout,
                           on_answer=(lambda name, answer: progress("answer", (name, answer)))
                           if progress else None, parallel=args.parallel)
              if questions else {"answers": {}})
    result["backend"] = "Gemma vision"
    if names is not None:
        result["shortlist"] = names
        result["observation"] = observation
    result["elapsed_s"] = time.perf_counter() - started
    return result


def print_result(result, threshold, as_json=False, timestamp=None, detail=True):
    shown = render(result, threshold)
    if as_json:
        data = {**shown, **{key: result[key] for key in ("backend", "shortlist", "observation", "elapsed_s")
                            if key in result}}
        print(json.dumps({"time": timestamp, **data} if timestamp else data), flush=True)
        return
    elapsed = result.get("elapsed_s")
    header = f"[{timestamp}] " if timestamp else ""
    header += result.get("backend", "Decision")
    if elapsed is not None:
        header += f" | {elapsed:.1f}s"
    print(header, flush=True)
    if detail and "shortlist" in result:
        print("  Vision shortlist: " + (", ".join(result["shortlist"]) or "none"), flush=True)
        observation = result.get("observation", {})
        print(f"  Scene: {observation.get('summary', '')} | {observation.get('setting', 'unclear')}, "
              f"{observation.get('lighting', 'unclear')} light", flush=True)
    for name, answer in (result["answers"].items() if detail else ()):
        if answer.get("type") == "binary":
            print(f"  Model score: Is {name} visible? {answer['probability']:.0%} yes", flush=True)
        elif answer.get("type") == "choice":
            print(f"  Model score: {name} = {answer['choice']}", flush=True)
        elif answer.get("type") == "score":
            print(f"  Model score: {name} = {answer['score']:.2f}", flush=True)
    objects = ", ".join(f"{item['name']} {item['probability']:.0%}" for item in shown["objects"])
    scene = ", ".join(f"{name}: {value}" for name, value in shown["scene"].items())
    print(f"  Visible: {objects or 'none above threshold'}"
          + (f" | {scene}" if scene else "") + "\n", flush=True)


def webcam(args, state, questions):
    try:
        import cv2
    except ImportError as exc:
        raise DecisionError("Install the webcam extra: uv sync --extra webcam") from exc
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise DecisionError(f"Could not open camera {args.camera}")
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    preview = args.preview and not args.no_preview
    print("Camera open. Gemma 3 4B will examine frames. Ctrl-C to stop."
          + (" Esc also closes the preview." if preview else ""), flush=True)
    pending = None
    events = queue.SimpleQueue()
    last_submit = 0.0
    last_heartbeat = 0.0
    frame_number = 0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise DecisionError("Could not read webcam frame")
            if preview:
                cv2.imshow("Visual Classifier (Esc to quit)", frame)
                if cv2.waitKey(1) == 27:
                    break
            while True:
                try:
                    kind, payload = events.get_nowait()
                except queue.Empty:
                    break
                if args.json:
                    continue
                if kind == "observation":
                    print("  Vision shortlist: " + (", ".join(payload["objects"]) or "none"), flush=True)
                    print(f"  Scene: {payload['summary']} | {payload['setting']}, {payload['lighting']} light", flush=True)
                elif kind == "answer":
                    name, answer = payload
                    if answer.get("type") == "binary":
                        print(f"  Model score: Is {name} visible? {answer['probability']:.0%} yes", flush=True)
                    else:
                        print(f"  Model score: {name} = {answer.get('choice', answer.get('score'))}", flush=True)
            if pending is not None and pending.done():
                result = pending.result()
                print_result(result, args.threshold, args.json,
                             dt.datetime.now().isoformat(timespec="seconds"), detail=False)
                pending = None
            if pending is not None and not args.json and time.monotonic() - last_heartbeat >= 10:
                print("  Gemma is still processing this frame...", flush=True)
                last_heartbeat = time.monotonic()
            if pending is None and time.monotonic() - last_submit >= args.interval:
                height, width = frame.shape[:2]
                if max(height, width) > args.max_edge:
                    scale = args.max_edge / max(height, width)
                    frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if not ok:
                    raise DecisionError("Could not encode camera frame")
                uri = "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode()
                pending = concurrent.futures.Future()
                frame_number += 1
                if not args.json:
                    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] Frame {frame_number}: asking Gemma...", flush=True)
                last_heartbeat = time.monotonic()
                def score_frame(future=pending, image=uri):
                    try:
                        future.set_result(_decide(args, state, questions, [image],
                                                  lambda kind, payload: events.put((kind, payload))))
                    except BaseException as exc:
                        future.set_exception(exc)
                threading.Thread(target=score_frame, daemon=True).start()
                last_submit = time.monotonic()
    finally:
        camera.release()
        if preview:
            cv2.destroyAllWindows()


def build_parser():
    parser = argparse.ArgumentParser(prog="visual-classifier", description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1", help="Local llama.cpp API base")
    parser.add_argument("--model", help="Local server alias (default: local-vision)")
    parser.add_argument("--objects", default="all",
                        help="'all' discovers from 36 common objects; comma-separated names score each one")
    parser.add_argument("--list-objects", action="store_true")
    parser.add_argument("--scene", action="store_true", help="Add indoor/outdoor and brightness checks")
    parser.add_argument("--config", help="JSON file with custom questions; overrides --objects/--scene")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-candidates", type=int, default=4,
                        help="Maximum shortlisted objects to score per frame (default: 4)")
    parser.add_argument("--parallel", type=int, default=2,
                        help="Concurrent local object checks (default: 2)")
    parser.add_argument("--timeout", type=float, default=90)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", help="Classify a JPEG, PNG, or WebP file")
    source.add_argument("--webcam", action="store_true", help="Continuously classify camera frames")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--interval", type=float, default=1.0, help="Minimum seconds between frame requests (default: 1)")
    parser.add_argument("--max-edge", type=int, default=640, help="Resize webcam longest edge")
    parser.add_argument("--preview", action="store_true", help="Show OpenCV camera window (off by default)")
    parser.add_argument("--no-preview", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true", help="Print full JSON results")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.list_objects:
        print("\n".join(OBJECTS))
        return 0
    try:
        if not 0 <= args.threshold <= 1 or args.max_edge < 64 or args.interval < 0 or args.timeout <= 0 or not 1 <= args.max_candidates <= 8 or not 1 <= args.parallel <= 4:
            raise DecisionError("Check threshold (0..1), max-edge (>=64), interval (>=0), timeout (>0), max-candidates (1..8), and parallel (1..4)")
        questions, state = _questions(args)
        if args.webcam or not args.image:
            webcam(args, state, questions)
        else:
            images = [image_uri(args.image)] if args.image else []
            print_result(_decide(args, state, questions, images), args.threshold, args.json)
        return 0
    except (DecisionError, OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"visual-classifier: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
