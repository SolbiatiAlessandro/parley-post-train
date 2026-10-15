"""Turn finished games into a policy-gradient training file.

For every seat of every finished game: reward = Parley's placement score for
that seat (share of the other seats it finished ahead of, ties half; the
five seats of a table average 0.5) minus a thinking-length penalty. The
advantage is the reward minus the table mean (group = the 5 seats of one
game), optionally divided by the table's standard deviation. Every call the
seat made gets that advantage.

Each output line: {"prompt": exact text the model read, "segments": [{"text", "train"}],
"advantage", "group", "seat", "game", "cut", "thinking_tokens"}.
The "limited time" sentence the recorder injects when the thinking budget is
hit is not the model's own text, so it is a non-trained segment.

Usage: .venv/bin/python build_trajectories.py --out rl/iter1.jsonl runs/rl-iter1-game*/
"""

import argparse
import json
import statistics
from pathlib import Path

CUT_SENTENCE = ("\n\nConsidering the limited time by the user, "
                "I have to give the solution based on the thinking directly now.\n</think>\n\n")


def read_calls(run_dir):
    latest = {}
    for line in (run_dir / "calls.jsonl").read_text().splitlines():
        call = json.loads(line)
        latest[call["n"]] = call
    return [latest[n] for n in sorted(latest)]


def seat_rewards(run_dir, calls, length_penalty, budget, cut_penalty):
    results = json.loads((run_dir / "results.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    names = [p["name"] for p in config["players"]]
    by_slot = {}
    for call in calls:
        by_slot.setdefault(call["slot"], []).append(call)
    rows = {}
    for slot, name in enumerate(names):
        score = results["scores"][results["names"].index(name)]
        mine = [c for c in by_slot.get(slot, []) if c.get("status") == "done"]
        think = [c["output_tokens"] - max(len(c.get("output") or "") // 4, 1) for c in mine]
        mean_think = statistics.mean(think) if think else 0
        cuts = sum(1 for c in mine if c.get("thinking_cut"))
        penalty = length_penalty * max(0.0, mean_think - budget) / 1000 + cut_penalty * (cuts / max(len(mine), 1))
        rows[slot] = {"score": score, "reward": score - penalty, "mean_thinking_tokens": mean_think,
                      "cut_fraction": cuts / max(len(mine), 1), "calls": mine}
    return rows


def segments_for(call):
    reasoning = call.get("reasoning") or ""
    output = call.get("output") or ""
    if call.get("thinking_cut"):
        return [{"text": "<think>\n" + reasoning, "train": True},
                {"text": CUT_SENTENCE, "train": False},
                {"text": output, "train": True}]
    if reasoning:
        return [{"text": "<think>\n" + reasoning + "\n</think>\n\n" + output, "train": True}]
    return [{"text": output, "train": True}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--length-penalty", type=float, default=0.1, help="reward lost per 1,000 mean thinking tokens over the budget")
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--cut-penalty", type=float, default=0.1, help="reward lost when every call hit the hard cutoff")
    parser.add_argument("--std-normalize", action="store_true", help="divide by the table's std (GRPO); default is Dr. GRPO style, no division")
    parser.add_argument("--train-model", default=None, help="only seats served by this model name (mixed tables)")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = n_games = 0
    summary = []
    with out.open("w") as stream:
        for run in args.runs:
            run_dir = Path(run)
            if not (run_dir / "results.json").exists():
                print(f"skip {run_dir.name}: no results")
                continue
            calls = read_calls(run_dir)
            rows = seat_rewards(run_dir, calls, args.length_penalty, args.budget, args.cut_penalty)
            rewards = [r["reward"] for r in rows.values()]
            mean = statistics.mean(rewards)
            std = statistics.pstdev(rewards) if args.std_normalize else 1.0
            n_games += 1
            for slot, row in rows.items():
                adv = (row["reward"] - mean) / (std + 1e-6)
                summary.append({"game": run_dir.name, "slot": slot, "score": row["score"], "reward": row["reward"],
                                "advantage": adv, "mean_thinking_tokens": row["mean_thinking_tokens"],
                                "cut_fraction": row["cut_fraction"], "calls": len(row["calls"])})
                for call in row["calls"]:
                    if args.train_model and call.get("served_model") != args.train_model:
                        continue
                    if not call.get("template_text"):
                        continue
                    stream.write(json.dumps({
                        "prompt": call["template_text"], "segments": segments_for(call),
                        "advantage": adv, "group": run_dir.name, "seat": call["seat"], "game": run_dir.name,
                        "cut": bool(call.get("thinking_cut")), "thinking_tokens": call["output_tokens"],
                    }) + "\n")
                    n_rows += 1
    (out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=1))
    print(f"{n_games} games, {n_rows} training calls -> {out}")
    if summary:
        print("mean score", round(statistics.mean(s["score"] for s in summary), 3),
              "mean thinking tokens", round(statistics.mean(s["mean_thinking_tokens"] for s in summary)),
              "cut fraction", round(statistics.mean(s["cut_fraction"] for s in summary), 3))


if __name__ == "__main__":
    main()
