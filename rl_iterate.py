"""One RL iteration: rollouts -> trajectories -> LoRA update -> serve the new adapter.

  .venv/bin/python rl_iterate.py --iteration 1 --games 8            # trains iter0 -> iter1
  .venv/bin/python rl_iterate.py --iteration 1 --skip-rollouts      # rollouts already played
  .venv/bin/python rl_iterate.py --evaluate 3 --games 10            # iter3 seat vs 4 base seats

Rollouts for iteration k are played by adapter iter(k-1) in self-play (five
seats, one policy); the update produces iter(k). Everything is logged to
rl/iterK.*.json and the games appear on the landing page as rl-iterK-gNN.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
PY = str(HERE / ".venv" / "bin" / "python")
OPERATOR_NOTE = "Keep your thinking short: under 1000 tokens of thinking, then answer."


def sh(cmd, log_name, env=None):
    print(time.strftime("%H:%M:%S"), " ".join(str(c) for c in cmd), flush=True)
    with (HERE / "logs" / log_name).open("a") as log:
        code = subprocess.call([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT, env=env)
    if code != 0:
        raise SystemExit(f"{cmd[0]} failed with exit {code}; see logs/{log_name}")


def rollouts(prefix, games, model, budget, trained=None, seed=None):
    cmd = [PY, HERE / "rl_rollouts.py", "--prefix", prefix, "--games", games, "--thinking",
           "--thinking-budget", budget, "--operator-note", OPERATOR_NOTE, "--model", model]
    if trained:
        cmd += ["--trained", trained, "--rotate-slot"]
    if seed is not None:
        cmd += ["--seed", seed]
    sh(cmd, f"rollouts-{prefix}.log")
    games_file = HERE / "rl" / f"{prefix}.games.json"
    summaries = json.loads(games_file.read_text())
    return [s for s in summaries if s["finished"]], summaries


def train(iteration, data_file, args):
    prev, new = f"iter{iteration - 1}", f"iter{iteration}"
    sh(["modal", "volume", "put", "parley-rl-data", HERE / data_file, data_file, "--force"], "modal-volume.log")
    sh(["modal", "run", str(HERE / "modal_train.py") + "::train_step", "--prev", prev, "--new", new,
        "--data-file", data_file, "--lr", args.lr, "--epochs", args.epochs, "--clip", args.clip,
        "--clip-high", args.clip_high, "--kl-coef", args.kl_coef, "--minibatches", args.minibatches,
        "--loss-agg", args.loss_agg], f"train-{new}.log")
    return new


def serve(adapters):
    """Point the LoRA server at these adapters and redeploy it (new containers reload the volume)."""
    active = HERE / "rl" / "active.json"
    active.write_text(json.dumps({"adapters": adapters}))
    sh(["modal", "volume", "put", "parley-adapters", active, "active.json", "--force"], "modal-volume.log")
    sh(["modal", "deploy", HERE / "modal_qwen.py"], "modal-deploy-lora.log", env={**__import__("os").environ, "PARLEY_APP": "parley-qwen-lora"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--evaluate", type=int, help="adapter iteration to evaluate against base")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--thinking-budget", type=int, default=1500)
    parser.add_argument("--skip-rollouts", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--clip-high", type=float, default=0.28)
    parser.add_argument("--kl-coef", type=float, default=0.0)
    parser.add_argument("--minibatches", type=int, default=4, help="optimizer steps per iteration")
    parser.add_argument("--loss-agg", default="token")
    parser.add_argument("--length-penalty", type=float, default=0.1)
    parser.add_argument("--cut-penalty", type=float, default=0.0)
    parser.add_argument("--std-normalize", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--served-name", default=None,
                        help="model name the rollouts were served under, if not iter(k-1) (iteration 1 played on the base server as qwen3.5-4b)")
    args = parser.parse_args()
    (HERE / "rl").mkdir(exist_ok=True)

    if args.evaluate is not None:
        name = f"iter{args.evaluate}"
        prefix = f"eval-{name}"
        finished, all_games = rollouts(prefix, args.games, "qwen3.5-4b", args.thinking_budget, trained=name,
                                       seed=args.seed)
        trained_scores, base_scores = [], []
        for i, g in enumerate(all_games):
            if not g["finished"]:
                continue
            slot = i % 5
            config = json.loads((HERE / "runs" / g["game"] / "config.json").read_text())
            names = [p["name"] for p in config["players"]]
            for k, n in enumerate(names):
                (trained_scores if k == slot else base_scores).append(g["scores"][n])
        report = {"adapter": name, "games_finished": len(finished), "games_total": len(all_games),
                  "trained_mean_score": statistics.mean(trained_scores) if trained_scores else None,
                  "trained_scores": trained_scores,
                  "base_mean_score": statistics.mean(base_scores) if base_scores else None,
                  "wins_by_trained_seat": sum(1 for s in trained_scores if s == 1.0)}
        if len(trained_scores) > 1:
            report["trained_score_stderr"] = statistics.stdev(trained_scores) / len(trained_scores) ** 0.5
        (HERE / "rl" / f"{prefix}.report.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report, indent=1))
        return

    k = args.iteration
    prefix = f"rl-iter{k}"
    policy = f"iter{k - 1}"
    if not args.skip_rollouts:
        finished, _ = rollouts(prefix, args.games, policy, args.thinking_budget, seed=args.seed)
        if len(finished) < max(2, args.games // 2):
            raise SystemExit(f"only {len(finished)} games finished; not training on so few")
    data_file = f"rl/{prefix}.jsonl"
    cmd = [PY, HERE / "build_trajectories.py", "--out", data_file, "--length-penalty", args.length_penalty,
           "--cut-penalty", args.cut_penalty, "--train-model", args.served_name or policy] + sorted(str(p) for p in (HERE / "runs").glob(f"{prefix}-g*"))
    if args.std_normalize:
        cmd.append("--std-normalize")
    sh(cmd, f"build-{prefix}.log")
    new = train(k, data_file, args)
    serve(["iter0", new] if new != "iter0" else ["iter0"])
    print(time.strftime("%H:%M:%S"), f"iteration {k} done: adapter {new} is now served", flush=True)


if __name__ == "__main__":
    main()
