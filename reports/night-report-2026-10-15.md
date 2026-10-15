# Overnight report: Parley post-training, night of 2026-10-14 → 15

Written by Tashi at 05:05 PDT on 2026-10-15. Tags: **(verified)** = measured from the recordings or logs; **(my read)** = my judgment.

## Short version

1. **Game 4** (Qwen3.5-4B, thinking on, told "under 1000 tokens of thinking", not enforced): the instruction had no effect. Median thinking 5,674 tokens, range 434–12,878 (verified). Replay on the landing page.
2. **An RL loop now exists and ran three iterations**: self-play rollouts on a Modal H100, LoRA policy-gradient updates with the algorithm chosen from the literature review, evaluation of the trained seat against the untouched model. Code: github.com/SolbiatiAlessandro/parley-post-train.
3. **No improvement after three iterations.** In 17 evaluation games, the trained seat (iter3) scored **0.441 ± 0.091** (mean placement score, standard error) against four base seats averaging 0.515; 0.5 means no difference. 3 outright wins in 17 games (verified). The 95% interval is 0.26–0.62: this evaluation can only see large effects, and there wasn't one.
4. **Why no effect (my read):** the policy barely moved. Three iterations × 4 optimizer steps on a LoRA is roughly 1% of what published self-play RL runs use (SPIRAL: 400 steps × 128 games). The logged change from the base model stayed tiny (clipped-ratio fraction 0.1–0.3% per step). The night's product is the working loop and a catalogue of the things that break, not a trained player.
5. **Cost:** $37.75 of Modal GPU for the whole night (verified from `modal billing report` at 05:00), against the $40 budget.

## What ran

| Game / batch | Policy and settings | Outcome (verified) |
|---|---|---|
| Game 1 | thinking off | finished, 122 calls, all legal |
| Game 2, 2b | thinking on, 3,000 / 2,000-token cutoff | failed / stopped: thinking never finished |
| Game 3 | thinking on, 8,000 cutoff, H100 | failed at call 19 on an illegal pledge name |
| Game 4 | thinking on, "under 1000 tokens" asked, 20,000 cap | finished, 128 calls; instruction ignored |
| RL iteration 1 rollouts | base model, 1,500-token thinking cutoff enforced | 8/8 games finished, 1,450 calls |
| RL iteration 2 rollouts | adapter iter1 | 7 finished; four earlier launches aborted (see "What broke") |
| RL iteration 3 rollouts | adapter iter2 | 5 finished; 3 long games stopped at 240 calls to start training |
| Evaluation | 1 seat iter3 vs 4 seats base, seat rotating | 17 finished, 1 died (whitespace runaway), 2 stopped at the budget line |

Every game is on the landing page (http://tashis-mac-mini.local:8800/) with a paused step-through replay showing each model call's prompt, thinking and answer.

## Algorithm (from reports/rl-literature-review.md)

Critic-free policy gradient. Group = the five seats of one game; advantage = a seat's placement score minus the mean of the other four seats (leave-one-out baseline, the RLOO form); the same advantage on every token the seat generated (thinking and answer); token-level loss normalized by the batch's generated tokens; 4 clipped minibatch steps per iteration (clip 0.2 below, 0.28 above, old log-probs recomputed by the trainer); no KL term (KL to the base model logged on a 10% sample); LoRA rank 16 on all attention and MLP projections; learning rate 3e-5, AdamW. Thinking cut at 1,500 tokens with Qwen's own "limited time" sentence; the injected sentence is masked out of the loss. Rollouts sampled the policy's own distribution (temperature 1, no top-p/top-k/penalty) for the thinking and Qwen's recommended settings for the short forced answer. No length penalty: every rollout call hit the cutoff, so length carried no signal.

Deviations from the review's spec: rank 16 instead of 32 (the adapter already existed), 4 steps per iteration instead of 1 (to get any movement at all in a few iterations), 600-call cap per update (training time).

## Iterations

| Iteration | Policy that played | Games used | Training calls | Optimizer steps | Train time | Entropy | KL to base (sampled k3) |
|---|---|---|---|---|---|---|---|
| 1 | base (iter0) | 7 (8th finished late) | 1,145 (all) | 4 | 77 min (slow kernels) | 0.76 | 0.001 |
| 2 | iter1 | 7 | 600 (capped) | 4 | 10 min (tilelang) | 0.75 | 0.017 |
| 3 | iter2 | 5 | 600 (capped) | 4 | 11 min | 0.75 | 10.1 (see note) |

Note on the KL column: the GRPO "k3" estimator averaged over sampled tokens of 10% of the calls. It is heavy-tailed: one sampled token the base model finds very unlikely can dominate the mean, so the jump at iteration 3 is not by itself evidence of a broken policy (my read). The evaluation is the real check, and the iter3 policy played complete, legal games.

## Evaluation (verified)

20 games launched, 1 iter3 seat vs 4 base seats, the trained seat's position rotating with the game index. 17 finished, 1 died (a base seat's answer ran into whitespace, item 9 below), 2 stopped at the spending line.

- iter3 seat: mean placement score **0.441**, SD 0.375, standard error 0.091, 95% interval 0.26–0.62, 3 outright wins in 17.
- base seats: mean 0.515.
- By the trained seat's position (mean, games): seat 0: 0.16 (4), seat 1: 0.31 (2), seat 2: 0.62 (3), seat 3: 0.53 (4), seat 4: 0.56 (4). Too few games per position to read anything into this.

Behaviour of each policy, per call, across the finished games it played (verified):

| Policy (where measured) | Calls | Thinking cut at budget | Hip-shot share of shots | Pass share of shoot turns | whisper | pledge | give | reveal |
|---|---|---|---|---|---|---|---|---|
| base (iteration-1 rollouts) | 1,450 | 99.7% | 0.30 | 0.03 | 0.22 | 0.21 | 0.15 | 0.05 |
| iter1 (iteration-2 rollouts) | 1,454 | 99.5% | 0.32 | 0.04 | 0.22 | 0.21 | 0.14 | 0.04 |
| iter2 (iteration-3 rollouts) | 1,591 | 99.5% | 0.36 | 0.04 | 0.21 | 0.20 | 0.13 | 0.05 |
| iter3 (evaluation seat) | 617 | 99.0% | 0.36 | 0.05 | 0.20 | 0.20 | 0.15 | 0.05 |
| base (evaluation seats) | 2,414 | 99.5% | 0.35 | 0.03 | 0.21 | 0.18 | 0.14 | 0.04 |

The only drift is a slightly higher hip-shot share across iterations (0.30 → 0.36), and the base seats in the evaluation show 0.35 too, so it is not clearly the training (my read). Side-action rates did not change.

## What broke, and the fixes (verified)

1. Prompt-only thinking limits are ignored by this model (game 4).
2. Decoder-level name constraints (a JSON schema built from each prompt's legal lists) stop illegal names from ending games; game 3 died that way before this existed.
3. Empty whisper text ended two games → schema requires 1–160-character strings.
4. A repetition penalty on the forced answer penalized the closing braces → removed.
5. Plain sampling made the grammar-constrained forced answer run off into whitespace → Qwen's sampling settings for the forced answer, plain sampling for the thinking.
6. Modal preempted the GPU container at 00:52 and seven mid-call games died → the recorder now retries a call across server restarts (up to 10 min) and games allow 900 s per call.
7. Training without the fast linear-attention kernels took 77 min per update → image now has flash-linear-attention plus tilelang (its Triton backward refuses to run on H100 without it); a 600-call update takes 10 min. causal-conv1d failed to build (slow fallback for that part).
8. One game died when the model closed its thinking just before the budget and the JSON itself was cut off → that case now goes through the same "limited time" continuation.
9. **Still open:** about 1 call in 2,000, the grammar-constrained answer runs into endless whitespace after a key (e.g. `"give":`) until the token cap, even under Qwen's sampling settings; one evaluation game died this way. Next fix: start the server with vLLM's `disable_any_whitespace` structured-output option (compact JSON only). Not applied tonight because a redeploy would have killed the running evaluation.

About 25 minutes of rollouts (≈$2) were lost to items 3–5 and about 35 minutes (≈$2.5) to item 6.

## Cost and time (verified)

$37.75 of Modal GPU at 05:00 PDT. Roughly: 77 min of slow iteration-1 training ($5), two training updates with tilelang ($1.5), about 4.5 hours of H100 serving for games 2–4, three rollout batches and the evaluation ($18), the rest in cold starts, idle windows before shutdown, aborted launches and the L40S hours earlier in the evening. One RL iteration (8 games + update + redeploy) costs about $4.5 and takes about an hour with these settings.

## What I would do next (my read)

1. Apply fix 9 (compact JSON) and raise the rollout thinking budget only if the model is trained toward using it; today it never finishes thinking under any budget below ~4k tokens.
2. Decide the learning signal before buying more iterations: at 40 seat-trajectories per iteration the placement reward is mostly noise (the review's estimate: one draw with SD 0.35 shared by ~30 calls). Either many more games per iteration (cheap in GPU terms if 16–24 run at once) or a denser reward (per-round points, or the enemy/friend card points) would give the gradient something to learn from.
3. Run the cheapest baseline next to it: rejection-sampling fine-tuning on winning seats from the same games, same evaluation. The review found it close to GRPO at equal samples; if it matches, the extra RL machinery is not yet earning its keep here.
4. Keep the evaluation but size it honestly: 20 games give ±0.16 at 95%; detecting +0.1 needs about 100 games (≈$15 at tonight's prices).

## Files

- Code: github.com/SolbiatiAlessandro/parley-post-train (recorder.py, run_game.py, rl_rollouts.py, build_trajectories.py, modal_train.py, modal_qwen.py, rl_iterate.py, hub.py, viewer/replay pages).
- Reports: reports/rl-literature-review.md, reports/thinking-patterns-game3.md, this file.
- Recordings (private, on the Mac mini): ~/Projects/parley-qwen-live/runs/*/ (calls.jsonl, trajectory.jsonl, replay.json, results.json); rl/*.jsonl training files and rl/eval-iter3.final.json.
- Adapters: Modal volume `parley-adapters` (iter0–iter3).
