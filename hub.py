"""Landing page for all recorded games, with a step-through replay of each.

Lists every run under runs/ that has recorded model calls. A finished game
(replay.json present) gets Parley's table replay next to the model calls;
a running or stopped game gets the model calls alone. Nothing here is live:
every page is paused and moves only when the viewer steps or presses play.

Usage: .venv/bin/python hub.py --port 8800
"""

import argparse
import asyncio
import json
import socket
import subprocess
from pathlib import Path

import aiohttp
from aiohttp import web

HERE = Path(__file__).parent
RUNS = HERE / "runs"
IMAGE = "parley-local:b46e7a8"


def read_calls(run_dir):
    latest = {}
    path = run_dir / "calls.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                call = json.loads(line)
            except ValueError:
                continue
            latest[call["n"]] = call
    return [latest[n] for n in sorted(latest)]


def summarize(run_dir):
    calls = read_calls(run_dir)
    if not calls:
        return None
    meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}
    results = None
    if (run_dir / "results.json").exists():
        results = json.loads((run_dir / "results.json").read_text())
    game = meta.get("game") or {}
    if results:
        state = "finished"
    elif game.get("exit") is not None:
        state = "failed"
    elif game.get("running") is False:
        state = "stopped"
    else:
        state = "running"
    done = [c for c in calls if c.get("status") == "done"]
    return {
        "name": run_dir.name,
        "state": state,
        "status": meta.get("status"),
        "model": meta.get("model"),
        "gpu": meta.get("gpu"),
        "thinking": meta.get("thinking", False),
        "thinking_budget": meta.get("thinking_budget"),
        "seed": game.get("seed"),
        "calls": len(calls),
        "cut": sum(1 for c in calls if c.get("thinking_cut")),
        "avg_latency_s": round(sum(c["latency_ms"] for c in done) / len(done) / 1000, 1) if done else None,
        "started": calls[0].get("started"),
        "last": calls[-1].get("started"),
        "winner": results["names"][results["win"].index(True)] if results and True in results["win"] else None,
        "scores": dict(zip(results["names"], results["rawScores"])) if results else None,
        "has_replay": (run_dir / "replay.json").exists(),
    }


class Hub:
    def __init__(self):
        self.session = None
        self.replay_ports = {}
        self.cache = {}

    async def start(self, _app):
        self.session = aiohttp.ClientSession()

    async def stop(self, _app):
        await self.session.close()

    def run_dir(self, request):
        name = request.match_info["run"]
        if "/" in name or name.startswith(".") or not (RUNS / name).is_dir():
            raise web.HTTPNotFound()
        return RUNS / name

    async def runs(self, _request):
        rows = [s for s in (summarize(d) for d in sorted(RUNS.iterdir()) if d.is_dir()) if s]
        rows.sort(key=lambda r: r["started"] or 0, reverse=True)
        return web.json_response(rows)

    async def index(self, _request):
        return web.FileResponse(HERE / "index.html")

    async def replay_page(self, _request):
        return web.FileResponse(HERE / "replay.html")

    async def replay_frame(self, _request):
        return web.FileResponse(HERE / "replay_frame.html")

    async def parley_client(self, request):
        name = request.match_info["name"]
        if name not in ("renderer.js", "chrome.css"):
            raise web.HTTPNotFound()
        return web.FileResponse(HERE / "upstream" / "client" / name)

    async def parley_asset(self, request):
        name = request.match_info["name"]
        if "/" in name or name.startswith("."):
            raise web.HTTPNotFound()
        return web.FileResponse(HERE / "upstream" / "data" / name)

    def replay_server_port(self, run_dir):
        """Port of a Parley replay server for this run, starting one if needed."""
        name = "parley-replay-" + run_dir.name
        if run_dir.name in self.replay_ports:
            return self.replay_ports[run_dir.name]
        shown = subprocess.run(["docker", "port", name], capture_output=True, text=True).stdout
        if "->" in shown:
            port = int(shown.strip().rsplit(":", 1)[-1])
        else:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            subprocess.run([
                "docker", "run", "-d", "--rm", "--name", name,
                "-p", f"127.0.0.1:{port}:{port}",
                "-v", f"{run_dir}:/run/parley",
                "-e", "COGAME_HOST=0.0.0.0", "-e", f"COGAME_PORT={port}",
                "-e", "COGAME_LOAD_REPLAY_URI=file:///run/parley/replay.json",
                IMAGE,
            ], check=True)
        self.replay_ports[run_dir.name] = port
        return port

    async def replay_data(self, request):
        """Replay payload (if the game finished) plus every recorded model call.

        Decisions are joined to calls by the exact user message the game sent."""
        run_dir = self.run_dir(request)
        calls = read_calls(run_dir)
        meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}
        payload, decisions = None, None
        if (run_dir / "replay.json").exists():
            cached = self.cache.get(run_dir.name)
            if cached:
                payload, decisions = cached
            else:
                port = self.replay_server_port(run_dir)
                for _ in range(50):
                    try:
                        async with self.session.ws_connect(f"ws://127.0.0.1:{port}/replay", max_msg_size=0) as ws:
                            payload = json.loads((await ws.receive()).data)
                        break
                    except aiohttp.ClientError:
                        await asyncio.sleep(0.3)
                if payload is None:
                    raise web.HTTPServiceUnavailable(text="replay server did not start")
                refs = json.loads((run_dir / "replay.json").read_text())["decisionRefs"]
                prompts = {}
                for line in (run_dir / "trajectory.jsonl").read_text().splitlines():
                    record = json.loads(line)
                    if record.get("event_type") != "decision":
                        continue
                    for attempt in record["attempts"]:
                        if attempt["attempt_id"] == record["selected_attempt_id"]:
                            prompts[str(record["decision_id"])] = attempt["request"]["messages"][0]["content"]
                by_user = {c["game_request"]["messages"][0]["content"]: c["n"] for c in calls}
                decisions = [{**ref, "n": by_user.get(prompts.get(str(ref["id"])))} for ref in refs]
                self.cache[run_dir.name] = (payload, decisions)
        return web.json_response({"payload": payload, "decisions": decisions,
                                  "calls": {c["n"]: c for c in calls}, "meta": meta,
                                  "summary": summarize(run_dir)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8800)
    args = parser.parse_args()
    hub = Hub()
    app = web.Application()
    app.on_startup.append(hub.start)
    app.on_cleanup.append(hub.stop)
    app.router.add_get("/", hub.index)
    app.router.add_get("/api/runs", hub.runs)
    app.router.add_get("/g/{run}/replay", hub.replay_page)
    app.router.add_get("/g/{run}/replay-frame", hub.replay_frame)
    app.router.add_get("/g/{run}/api/replay", hub.replay_data)
    app.router.add_get("/parley/client/assets/{name}", hub.parley_asset)
    app.router.add_get("/parley/client/{name}", hub.parley_client)
    web.run_app(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
