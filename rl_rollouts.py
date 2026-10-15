"""Play several Parley games at once, each with its own recorder and ports.

Used for RL rollouts (all seats one policy) and for evaluation (mixed tables:
--trained NAME --trained-slot K serves seat K from a LoRA adapter and the other
seats from the base model). Each game gets runs/<prefix>-gNN/ with the usual
recorder files, so the landing page lists every game.

Usage:
  .venv/bin/python rl_rollouts.py --prefix rl-iter1 --games 8 --thinking --thinking-budget 1500 \
      --operator-note "..." --model iter0
  .venv/bin/python rl_rollouts.py --prefix eval-iter3 --games 10 --thinking --thinking-budget 1500 \
      --trained iter3 --rotate-slot
"""

import argparse
import asyncio
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
PY = HERE / ".venv" / "bin" / "python"


def free_port(start, used):
    port = start
    while True:
        if port not in used:
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) != 0:
                    used.add(port)
                    return port
        port += 1


async def run_one(index, args, modal_url, used_ports):
    name = f"{args.prefix}-g{index:02d}"
    run_dir = HERE / "runs" / name
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    rec_port = free_port(9000 + index * 2, used_ports)
    game_port = free_port(9001 + index * 2, used_ports)
    rec_cmd = [str(PY), str(HERE / "recorder.py"), "--modal-url", modal_url, "--run-dir", str(run_dir),
               "--port", str(rec_port), "--parley-port", str(game_port), "--gpu", args.gpu,
               "--sampling", args.sampling]
    if args.thinking:
        rec_cmd += ["--thinking", "--thinking-budget", str(args.thinking_budget)]
    if args.trained:
        slot = (index % 5) if args.rotate_slot else args.trained_slot
        rec_cmd += ["--route", f"{slot}={args.trained}"]
        for other in range(5):
            if other != slot:
                rec_cmd += ["--route", f"{other}={args.model}"]
    else:
        for slot in range(5):
            rec_cmd += ["--route", f"{slot}={args.model}"]
    rec_log = (run_dir / "recorder.log").open("w")
    recorder = subprocess.Popen(rec_cmd, stdout=rec_log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", rec_port)) == 0:
                break
        await asyncio.sleep(0.3)
    game_cmd = [str(PY), str(HERE / "run_game.py"), "--modal-url", modal_url, "--run-dir", str(run_dir),
                "--recorder", f"http://127.0.0.1:{rec_port}", "--recorder-port", str(rec_port),
                "--parley-port", str(game_port), "--label", args.prefix]
    if args.thinking:
        game_cmd += ["--thinking"]
    if args.operator_note:
        game_cmd += ["--operator-note", args.operator_note]
    if args.seed is not None:
        game_cmd += ["--seed", str(args.seed + index)]
    game_log = (run_dir / "runner.log").open("w")
    started = time.monotonic()
    game = await asyncio.create_subprocess_exec(*game_cmd, stdout=game_log, stderr=subprocess.STDOUT)
    code = await game.wait()
    recorder.terminate()
    results = run_dir / "results.json"
    summary = {"game": name, "exit": code, "seconds": round(time.monotonic() - started),
               "finished": results.exists()}
    if results.exists():
        r = json.loads(results.read_text())
        summary["scores"] = dict(zip(r["names"], r["scores"]))
    print(json.dumps(summary), flush=True)
    return summary


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--thinking-budget", type=int, default=1500)
    parser.add_argument("--operator-note", default="")
    parser.add_argument("--model", default="qwen3.5-4b", help="served model/adapter name for ordinary seats")
    parser.add_argument("--trained", default=None, help="adapter name for the trained seat (evaluation)")
    parser.add_argument("--trained-slot", type=int, default=0)
    parser.add_argument("--rotate-slot", action="store_true", help="trained seat = game index mod 5")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu", default="H100")
    parser.add_argument("--sampling", choices=["qwen", "rl"], default="rl")
    parser.add_argument("--stagger", type=float, default=2.0, help="seconds between game launches")
    args = parser.parse_args()
    modal_url = (HERE / "logs" / "modal-url").read_text().strip()
    used = set()
    tasks = []
    for i in range(args.games):
        tasks.append(asyncio.create_task(run_one(i, args, modal_url, used)))
        await asyncio.sleep(args.stagger)
    summaries = await asyncio.gather(*tasks)
    finished = [s for s in summaries if s["finished"]]
    print(f"{len(finished)}/{len(summaries)} games finished", file=sys.stderr)
    (HERE / "rl" / f"{args.prefix}.games.json").write_text(json.dumps(summaries, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
