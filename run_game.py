"""Play one five-seat Parley game where every seat is Qwen3.5-4B.

Order: wake the Modal GPU and wait until vLLM answers, start the Parley game
container, connect five player sockets that each register the same operator
prompt, then wait for the final scores. The recorder (recorder.py) must
already be running; it carries every model call and shows them live.

Usage: .venv/bin/python run_game.py --modal-url URL --run-dir runs/<id>
"""

import argparse
import asyncio
import json
import secrets
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import aiohttp
import websockets
from dotenv import dotenv_values

HERE = Path(__file__).parent
IMAGE = "parley-local:b46e7a8"
# parley_player.nim's DefaultPrompt: what a seat plays with when PLAYER_PROMPT is empty.
OPERATOR_PROMPT = """
Play to win, but make it fun. Your secret cards run the round: steer shots
toward your ENEMY without being obvious about it (land at least one hit on
them yourself to earn its point), quietly keep your FRIEND alive, and never
reveal either card out loud. Shoot whoever threatens you or your friend
most, and aim for the head when you mean it. Shoot from the hip when you want the gun
to move without the damage - handing it to your friend, or staging a grudge
the table will believe - since nobody learns how you aimed, only whether it
landed. Keep your table talk short, funny, and a little scheming - propose
truces you may or may not honor, and let the table do your dirty work when
it will. Whisper to firm up a private deal, show a card when proof would win
you an ally, pledge when a public promise buys you safety, and pay a point
when a bribe is cheaper than a loss.
"""


async def status(session, recorder, **fields):
    print(time.strftime("%H:%M:%S"), json.dumps(fields)[:300], flush=True)
    async with session.post(recorder + "/api/status", json=fields) as response:
        response.raise_for_status()


async def wake_gpu(session, recorder, modal_url, key):
    await status(session, recorder, status="waking the GPU on Modal (first start downloads the model)",
                 gpu_started=time.time())
    began = time.monotonic()
    while time.monotonic() - began < 25 * 60:
        try:
            async with session.get(modal_url + "/v1/models",
                                   headers={"Authorization": f"Bearer {key}"},
                                   timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 200:
                    await status(session, recorder,
                                 status=f"GPU ready after {time.monotonic() - began:.0f} s")
                    return
                print("GPU not ready:", response.status, flush=True)
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            print("GPU not ready:", type(error).__name__, flush=True)
        await asyncio.sleep(10)
    raise SystemExit("GPU did not become ready within 25 minutes")


def require_free_port(port):
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(f"port {port} is already in use by another program")


def wait_for_server(port, seconds=60):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.3)
    raise SystemExit("Parley server did not become healthy")


async def play_seat(port, slot, token, finals):
    url = f"ws://127.0.0.1:{port}/player?slot={slot}&token={token}"
    # No client pings: a seat only listens, and a missed pong would count as a
    # disconnect, which ends the whole game.
    async with websockets.connect(url, max_size=None, ping_interval=None) as ws:
        registered = False
        async for raw in ws:
            message = json.loads(raw)
            kind = message.get("type")
            if kind == "welcome" and not registered:
                await ws.send(json.dumps({"type": "prompt", "prompt": OPERATOR_PROMPT, "scripted": False}))
                registered = True
                print(f"slot {slot} is {message['name']}; operator prompt delivered", flush=True)
            elif kind == "final":
                finals[slot] = message
                return


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modal-url", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--recorder", default="http://127.0.0.1:8790")
    parser.add_argument("--recorder-port", type=int, default=8790)
    parser.add_argument("--parley-port", type=int, default=8890)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--thinking", action="store_true",
                        help="thinking mode: Qwen's temperature 1.0 and longer deadlines")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    key = dotenv_values(HERE / ".env")["VLLM_API_KEY"]
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    config = {
        "seed": seed,
        "tokens": [secrets.token_hex(12) for _ in range(5)],
        "players": [{"name": f"qwen3.5-4b{'-think' if args.thinking else ''}-{i + 1}"} for i in range(5)],
        "model": "qwen/qwen3.5-4b",
        # Pacing for a watched local game (not the hosted defaults): slower
        # turns, a longer model deadline and episode budget. Parley deals a
        # new round only while the longest round so far fits in 60% of the
        # episode budget, so a slow thinking game needs a large budget.
        "turnDelayMs": 2500,
        "llmTimeoutSeconds": 300 if args.thinking else 90,
        "episodeTimeoutSeconds": 10800 if args.thinking else 2400,
    }
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=1))
    config_path.chmod(0o600)

    async with aiohttp.ClientSession() as session:
        require_free_port(args.parley_port)
        await wake_gpu(session, args.recorder, args.modal_url.rstrip("/"), key)
        name = "parley-live-" + run_dir.name
        subprocess.run([
            "docker", "run", "-d", "--rm", "--name", name,
            "-p", f"0.0.0.0:{args.parley_port}:{args.parley_port}",
            "-v", f"{run_dir}:/run/parley",
            "-e", "COGAME_HOST=0.0.0.0", "-e", f"COGAME_PORT={args.parley_port}",
            "-e", "COGAME_CONFIG_URI=file:///run/parley/config.json",
            "-e", "COGAME_RESULTS_URI=file:///run/parley/results.json",
            "-e", "COGAME_SAVE_REPLAY_URI=file:///run/parley/replay.json",
            "-e", "COGAME_SAVE_TRAJECTORY_URI=file:///run/parley/trajectory.jsonl",
            "-e", f"COWORLD_LLM_ENDPOINT=http://host.docker.internal:{args.recorder_port}",
            "-e", "COWORLD_LLM_MODEL=qwen/qwen3.5-4b",
            "-e", "COWORLD_LLM_TEMPERATURE=" + ("1.0" if args.thinking else "0.7"),
            "-e", "LLM_REQUEST_METADATA=" + json.dumps({"episode_request_id": "local-" + run_dir.name}),
            IMAGE,
        ], check=True)
        logger = subprocess.Popen(["docker", "logs", "-f", name],
                                  stdout=(run_dir / "game.log").open("w"), stderr=subprocess.STDOUT)
        wait_for_server(args.parley_port)
        await status(session, args.recorder, status="game running", game={"running": True, "seed": seed})
        finals = {}
        try:
            await asyncio.gather(*(play_seat(args.parley_port, slot, token, finals)
                                   for slot, token in enumerate(config["tokens"])))
        finally:
            code = subprocess.run(["docker", "wait", name], capture_output=True, text=True).stdout.strip()
            logger.wait(timeout=10)
        results = run_dir / "results.json"
        summary = json.loads(results.read_text()) if results.exists() else None
        replay = False
        if (run_dir / "replay.json").exists():
            # Serve the finished game's replay on the same port so /replay works.
            subprocess.run([
                "docker", "run", "-d", "--rm", "--name", "parley-replay-" + run_dir.name,
                "-p", f"0.0.0.0:{args.parley_port}:{args.parley_port}",
                "-v", f"{run_dir}:/run/parley",
                "-e", "COGAME_HOST=0.0.0.0", "-e", f"COGAME_PORT={args.parley_port}",
                "-e", "COGAME_LOAD_REPLAY_URI=file:///run/parley/replay.json",
                IMAGE,
            ], check=True)
            replay = True
        await status(session, args.recorder,
                     status=f"game over (exit {code})" + ("" if summary else " — no results: see game.log"),
                     game={"running": True, "seed": seed, "exit": code, "results": summary, "replay": replay},
                     gpu_stopped=time.time() + 600)


if __name__ == "__main__":
    asyncio.run(main())
