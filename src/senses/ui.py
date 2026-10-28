"""The read-only history page served by the daemon at /."""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>senses history</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #0f1115; color: #e6e8ee; font: 15px/1.5 system-ui, sans-serif; }
  header { display: flex; align-items: baseline; gap: 14px; padding: 18px 22px 10px; }
  h1 { margin: 0; font-size: 20px; letter-spacing: .3px; }
  .muted { color: #8a90a0; font-size: 13px; }
  form { display: flex; flex-wrap: wrap; gap: 8px; padding: 0 22px 14px; }
  input, select, button { background: #1a1e27; color: #e6e8ee; border: 1px solid #2a3040; border-radius: 7px;
    padding: 8px 10px; font: inherit; }
  input[type=search] { flex: 1 1 260px; min-width: 200px; }
  input:focus, select:focus, button:focus { outline: 2px solid #3d6ef7; outline-offset: 1px; }
  button { cursor: pointer; }
  button:hover { border-color: #3d6ef7; }
  label.live { display: inline-flex; align-items: center; gap: 6px; color: #8a90a0; font-size: 13px; }
  main { padding: 0 22px 30px; }
  .event { display: grid; grid-template-columns: 92px 150px 1fr; gap: 12px; padding: 9px 12px;
    border-top: 1px solid #1c212b; }
  .event:first-child { border-top: 0; }
  .event:hover { background: #141821; }
  .ts { color: #8a90a0; font-variant-numeric: tabular-nums; font-size: 13px; white-space: nowrap; }
  .src { color: #9db4ff; font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .text { white-space: pre-wrap; overflow-wrap: anywhere; }
  .text .kind { display: inline-block; margin-right: 8px; padding: 0 6px; border-radius: 4px;
    background: #202634; color: #8a90a0; font-size: 11px; vertical-align: 1px; }
  .empty { color: #8a90a0; padding: 28px 12px; }
  footer { position: sticky; bottom: 0; padding: 8px 22px; background: #0f1115ee; color: #8a90a0; font-size: 12px; }
  @media (max-width: 640px) {
    .event { grid-template-columns: 78px 1fr; }
    .src { grid-column: 2; }
    .text { grid-column: 2; }
  }
</style>
</head>
<body>
<header>
  <h1>senses history</h1>
  <span id="status" class="muted">connecting...</span>
</header>
<form id="filters">
  <input id="q" type="search" placeholder="Search the history, e.g. deployment or error" autocomplete="off">
  <select id="range" aria-label="Time range">
    <option value="15m">last 15 min</option>
    <option value="1h" selected>last hour</option>
    <option value="6h">last 6 hours</option>
    <option value="24h">last 24 hours</option>
    <option value="7d">last 7 days</option>
    <option value="all">everything</option>
  </select>
  <select id="source" aria-label="Source">
    <option value="">all sources</option>
    <option value="screen">screen</option>
    <option value="camera">camera</option>
    <option value="mic">microphone</option>
    <option value="desktop">desktop audio</option>
    <option value="presence">presence</option>
  </select>
  <button type="submit">Load</button>
  <label class="live"><input id="live" type="checkbox" checked> live</label>
</form>
<main id="events"><p class="empty">Loading...</p></main>
<footer id="meta"></footer>
<script>
const $ = (id) => document.getElementById(id);
const ALL_TIME = "1970-01-01T00:00:00";
let inFlight = false;

const pad = (n) => String(n).padStart(2, "0");
function stamp(ts) {
  const d = new Date(ts * 1000);
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  const time = pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds());
  return sameDay ? time : (d.getMonth() + 1) + "-" + pad(d.getDate()) + " " + time;
}

function addEvent(event) {
  const row = document.createElement("div");
  row.className = "event";

  const ts = document.createElement("div");
  ts.className = "ts";
  ts.textContent = stamp(event.ts);

  const source = document.createElement("div");
  source.className = "src";
  source.textContent = event.source || "-";
  source.title = event.source || "";

  const text = document.createElement("div");
  text.className = "text";
  if (event.kind && !event.kind.startsWith("presence.")) {
    const kind = document.createElement("span");
    kind.className = "kind";
    kind.textContent = event.kind;
    text.appendChild(kind);
  }
  text.appendChild(document.createTextNode(event.text || ""));
  row.append(ts, source, text);
  return row;
}

function show(events) {
  const main = $("events");
  main.replaceChildren();
  if (!events.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = $("q").value.trim() ? "No matches." : "No events in this range.";
    main.appendChild(empty);
  } else {
    const frag = document.createDocumentFragment();
    events.forEach((event) => frag.appendChild(addEvent(event)));
    main.appendChild(frag);
  }
  $("meta").textContent = events.length + (events.length === 1 ? " event" : " events")
    + ($("q").value.trim() ? " matching " + JSON.stringify($("q").value.trim()) : "");
}

async function load() {
  if (inFlight) return;
  inFlight = true;
  const query = $("q").value.trim();
  const range = $("range").value;
  const params = new URLSearchParams({
    since: range === "all" ? ALL_TIME : range,
    limit: query ? "200" : "1000",
  });
  if ($("source").value) params.set("sources", $("source").value);
  if (query) params.set("q", query);
  try {
    const response = await fetch((query ? "/control/search?" : "/control/timeline?") + params.toString());
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || response.statusText);
    show(body);
  } catch (err) {
    $("meta").textContent = "error: " + err.message;
  } finally {
    inFlight = false;
  }
}

async function status() {
  try {
    const response = await fetch("/control/status");
    const s = await response.json();
    $("status").textContent = "presence: " + s.presence + " | capture: " + s.capture.replace("_", " ");
  } catch (err) {
    $("status").textContent = "daemon unreachable";
  }
}

$("filters").addEventListener("submit", (event) => { event.preventDefault(); load(); });
$("range").addEventListener("change", load);
$("source").addEventListener("change", load);
let typing;
$("q").addEventListener("input", () => { clearTimeout(typing); typing = setTimeout(load, 300); });
document.addEventListener("visibilitychange", () => { if (!document.hidden) load(); });

setInterval(() => {
  if (!$("live").checked || $("q").value.trim() || document.hidden) return;
  load();
}, 5000);
setInterval(status, 10000);
status();
load();
</script>
</body>
</html>
"""
