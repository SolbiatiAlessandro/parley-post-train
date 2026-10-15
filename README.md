# parley-post-train

Post-training small open models on [Parley](https://github.com/Metta-AI/cogame-parley), a five-player
talk-and-shoot party game from Softmax's Coworld platform. Built on the night of 2026-10-14 → 15 by
Tashi (Alessandro Solbiati's agent) on a Mac mini, with the model served and trained on Modal GPUs.

**Start with the reports:**

- [`reports/night-report-2026-10-15.md`](reports/night-report-2026-10-15.md) — what ran, the numbers,
  nine failure modes with fixes, cost, next steps; appendices list every game and every training step.
- [`reports/rl-literature-review.md`](reports/rl-literature-review.md) — the review (60 sources) behind
  the RL algorithm choice, with a concrete spec.
- [`reports/thinking-patterns-game3.md`](reports/thinking-patterns-game3.md) — what Qwen3.5-4B does in
  its thinking during a game, from 19 full traces.

## Result in one paragraph

Qwen3.5-4B plays complete, legal games of Parley out of the box once the decoder is limited to legal
names. Its thinking mode is long (median ~6,000 tokens per call) and ignores prompt-level length requests.
Three iterations of self-play RL (LoRA, leave-one-out advantage over the five seats of a game, token-level
loss, 4 clipped steps per iteration) produced no measurable change: in 17 evaluation games one trained seat
against four untouched seats scored 0.441 ± 0.091 (mean placement, standard error) versus 0.515 for the
untouched seats. The policy barely moved; the budget ($40 of GPU) bought a working loop and a catalogue of
what breaks, not a trained player.

## How it fits together

```
Parley game (Docker, Mac)  --Anthropic Messages request-->  recorder.py (Mac)
recorder.py  --OpenAI chat request-->  vLLM serving Qwen (+ LoRA adapters) on a Modal GPU (modal_qwen.py)
recorder.py  --reply-->  game; every request/reply saved to runs/<game>/calls.jsonl
hub.py       --landing page-->  paused step-through replays of every recorded game
rl_iterate.py:  rl_rollouts.py (N games at once) -> build_trajectories.py -> modal_train.py (LoRA update) -> redeploy
```

| File | What it does |
|---|---|
| `modal_qwen.py` | Serves the model with vLLM on one Modal GPU. Starts on the first request, stops after 10 idle minutes. Serves LoRA adapters listed in `/adapters/active.json` on the `parley-adapters` volume. `PARLEY_APP=name modal deploy modal_qwen.py` deploys a second server. |
| `recorder.py` | Sits between the game and the GPU. Translates the game's request, builds a JSON schema per call so only legal names can be produced, streams the thinking to the live page, applies Qwen's thinking-budget cutoff, retries across server restarts, saves everything. `--route slot=adapter` serves one seat from another adapter (mixed tables). |
| `run_game.py` | Plays one game: wakes the GPU, starts the Parley container, connects five players with the operator prompt, waits for the scores, then starts a replay server for the finished game. |
| `viewer.html`, `replay.html`, `replay_frame.html` | Live page and paused step-through replay: Parley's own table view next to each model call's prompt, secret cards, thinking and answer. |
| `hub.py`, `index.html` | Landing page over `runs/`: every game, its state, and its replay. |
| `rl_rollouts.py` | Runs N games in parallel, each with its own recorder and ports; evaluation mode puts a trained adapter in one rotating seat. |
| `build_trajectories.py` | Turns finished games into training rows: per-seat reward (placement score minus an optional thinking-length penalty), leave-one-out advantage, every call of the seat, injected text masked. |
| `modal_train.py` | Modal functions: `make_adapter` (identity LoRA) and `train_step` (clipped policy-gradient update on a LoRA, token-level loss, optional KL, KL-to-base logging). |
| `rl_iterate.py` | One RL iteration end to end, or `--evaluate K` for the mixed-table evaluation. |

## Settings used

Qwen3.5-4B (`Qwen/Qwen3.5-4B`), vLLM 0.31.0, one H100 (L40S for the first game). Parley pinned at commit
`b46e7a8`, built locally, ordinary seed-sampled tables (3–6 rounds, 2–3 hp, 1–3 survivors). Games 1–4 used
Qwen's recommended sampling; RL rollouts sampled the policy's own distribution for the thinking and Qwen's
settings for the short forced answer. Thinking budget 1,500 tokens for rollouts (every call hit it).
Training: LoRA rank 16 on q,k,v,o,gate,up,down, lr 3e-5, 4 minibatch steps per iteration, clip 0.2/0.28, no
KL term, 600-call cap per update, flash-linear-attention + tilelang kernels.

## Run it

```bash
git clone https://github.com/Metta-AI/cogame-parley upstream && (cd upstream && git checkout b46e7a8)
docker build --build-arg PARLEY_SOURCE_REVISION=$(git -C upstream rev-parse HEAD) \
  --build-arg PARLEY_GAME_VERSION=0.4.0+local -t parley-local:b46e7a8 upstream
python3 -c "import secrets; print('VLLM_API_KEY=' + secrets.token_urlsafe(32))" > .env
uv venv -p 3.12 .venv && uv pip install -p .venv aiohttp websockets python-dotenv

modal deploy modal_qwen.py                      # base server; URL goes into logs/modal-url
.venv/bin/python hub.py --port 8800 &           # landing page
.venv/bin/python recorder.py --modal-url <url> --run-dir runs/game1 --thinking &
.venv/bin/python run_game.py --modal-url <url> --run-dir runs/game1 --thinking

# RL
modal run modal_train.py::make_adapter --name iter0
echo '{"adapters": ["iter0"]}' > rl/active.json && modal volume put parley-adapters rl/active.json active.json
PARLEY_APP=parley-qwen-lora modal deploy modal_qwen.py   # LoRA-serving server; put its URL in logs/modal-url
.venv/bin/python rl_iterate.py --iteration 1 --games 8 --served-name qwen3.5-4b
.venv/bin/python rl_iterate.py --iteration 2 --games 8
.venv/bin/python rl_iterate.py --evaluate 2 --games 20
```

`runs/` (game recordings) and `rl/` (training files) are private and not committed; small per-seat
summaries and the evaluation scores are in `results/`.

## Known open issue

About 1 call in 2,000, the grammar-constrained answer runs into endless whitespace after a key until the
token cap; the game then ends. Fix to apply next: vLLM's `disable_any_whitespace` structured-output option
on the server (compact JSON only).
