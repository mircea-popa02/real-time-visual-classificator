"""Single-command interface for still images, webcam, and text state."""

import argparse
import base64
import concurrent.futures
import datetime as dt
import json
import mimetypes
import os
from pathlib import Path
import sys
import threading
import time

from .catalog import OBJECTS, SCENE_QUESTIONS, object_questions
from .decision import DecisionError, jev_decide, local_decide, local_discover


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
    present = sorted(((name, answer["noul"]) for name, answer in answers.items()
                      if answer.get("type") == "noul" and answer.get("noul", 0) >= threshold),
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


def _decide(args, state, questions, images):
    started = time.perf_counter()
    if args.backend == "jev":
        if images:
            raise DecisionError("Jev's documented API does not accept image attachments; use --backend local")
        result = jev_decide(state, questions, args.model, args.jev_url, args.timeout)
        result["backend"] = "JEV API"
        result["elapsed_s"] = time.perf_counter() - started
        return result
    model = args.model or "local-vision"
    names = None
    if images and not args.config and args.objects == "all":
        names = local_discover(images, OBJECTS, args.url, model,
                               os.environ.get("LOCAL_API_KEY"), args.timeout)
        questions = object_questions(names) if names else {}
        if args.scene:
            questions.update(SCENE_QUESTIONS)
        if not questions:
            return {"answers": {}, "shortlist": [], "backend": "Gemma vision",
                    "elapsed_s": time.perf_counter() - started}
    result = local_decide(state, questions, images, args.url, model,
                          os.environ.get("LOCAL_API_KEY"), args.timeout)
    result["backend"] = "Gemma vision" if images else "local model"
    if names is not None:
        result["shortlist"] = names
    result["elapsed_s"] = time.perf_counter() - started
    return result


def print_result(result, threshold, as_json=False, timestamp=None):
    shown = render(result, threshold)
    if as_json:
        data = {**shown, **{key: result[key] for key in ("backend", "shortlist", "elapsed_s")
                            if key in result}}
        print(json.dumps({"time": timestamp, **data} if timestamp else data), flush=True)
        return
    elapsed = result.get("elapsed_s")
    header = f"[{timestamp}] " if timestamp else ""
    header += result.get("backend", "Decision")
    if elapsed is not None:
        header += f" | {elapsed:.1f}s"
    print(header, flush=True)
    if "shortlist" in result:
        print("  Vision shortlist: " + (", ".join(result["shortlist"]) or "none"), flush=True)
    for name, answer in result["answers"].items():
        if answer.get("type") == "noul":
            label = f"JEV: {name}" if result.get("backend") == "JEV API" else f"Jev-style: Is {name} visible?"
            print(f"  {label} {answer['noul']:.0%} yes", flush=True)
        elif answer.get("type") == "choice":
            print(f"  Jev-style: {name} = {answer['choice']}", flush=True)
        elif answer.get("type") == "score":
            print(f"  Jev-style: {name} = {answer['score']:.2f}", flush=True)
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
    print("Camera open. Gemma 3 4B will examine frames; Esc or Ctrl-C to stop.", flush=True)
    pending = None
    last_submit = 0.0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise DecisionError("Could not read webcam frame")
            if not args.no_preview:
                cv2.imshow("Jev Vision Probe (Esc to quit)", frame)
                if cv2.waitKey(1) == 27:
                    break
            if pending is not None and pending.done():
                result = pending.result()
                print_result(result, args.threshold, args.json,
                             dt.datetime.now().isoformat(timespec="seconds"))
                pending = None
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
                def score_frame(future=pending, image=uri):
                    try:
                        future.set_result(_decide(args, state, questions, [image]))
                    except BaseException as exc:
                        future.set_exception(exc)
                threading.Thread(target=score_frame, daemon=True).start()
                last_submit = time.monotonic()
    finally:
        camera.release()
        cv2.destroyAllWindows()


def build_parser():
    parser = argparse.ArgumentParser(prog="jev-vision", description=__doc__)
    parser.add_argument("--backend", choices=("local", "jev"), default="local")
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1", help="Local llama.cpp API base")
    parser.add_argument("--jev-url", default="https://www.jevai.org", help="Jev API origin")
    parser.add_argument("--model", help="Local server alias (default: local-vision); optional Jev model ID")
    parser.add_argument("--objects", default="all",
                        help="'all' discovers from 36 common objects; comma-separated names score each one")
    parser.add_argument("--list-objects", action="store_true")
    parser.add_argument("--scene", action="store_true", help="Add indoor/outdoor and brightness questions")
    parser.add_argument("--config", help="JSON file with state and questions; overrides --objects/--scene")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=90)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", help="Classify a JPEG, PNG, or WebP file")
    source.add_argument("--webcam", action="store_true", help="Continuously classify camera frames")
    source.add_argument("--state", help="Text/JSON state to decide with Jev or local model")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--interval", type=float, default=1.0, help="Minimum seconds between frame requests (default: 1)")
    parser.add_argument("--max-edge", type=int, default=640, help="Resize webcam longest edge")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print full JSON results")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.list_objects:
        print("\n".join(OBJECTS))
        return 0
    try:
        if not 0 <= args.threshold <= 1 or args.max_edge < 64 or args.interval < 0 or args.timeout <= 0:
            raise DecisionError("Check threshold (0..1), max-edge (>=64), interval (>=0), and timeout (>0)")
        questions, state = _questions(args)
        if args.backend == "jev" and (args.image or args.webcam or not (args.state or args.config)):
            raise DecisionError("Jev's documented API is text-only; choose --backend local for images")
        if args.webcam or not (args.image or args.state or args.config):
            webcam(args, state, questions)
        else:
            if args.state:
                try:
                    state = json.loads(args.state)
                except json.JSONDecodeError:
                    state = args.state
            images = [image_uri(args.image)] if args.image else []
            print_result(_decide(args, state, questions, images), args.threshold, args.json)
        return 0
    except (DecisionError, OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"jev-vision: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
