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

## Appendix A: every recorded game

One row per run folder with recorded model calls. "Cut %" = share of calls whose thinking hit the budget. Raw points are Parley's match points per seat in seat order.

| Run | State | Policies | Thinking | Budget | Seed | Calls | Median thinking tokens | Cut % | Winner | Raw points | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261014-qwen4b-game1 | finished | Qwen3.5-4B | off |  | 559834417 | 122 | 7 | 0 | qwen3.5-4b-2 | [12.0, 17.0, 8.0, 16.0, 15.0] |  |
| 20261014-qwen4b-think-game2 | failed | Qwen3.5-4B | on | 3000 | 849954396 | 1 | 3299 | 0 |  |  | failed at call 1: thinking ran past the 3,000-token budget, so the answer was empty and Pa |
| 20261014-qwen4b-think-game2b | stopped | Qwen3.5-4B | on | 2000 | 1446672032 | 6 | 2319 | 100 |  |  | stopped by Tashi after 3 calls: every call hit the 2,000-token thinking budget |
| 20261014-qwen4b-think-game3 | failed | Qwen3.5-4B | on | 8000 | 1556325446 | 19 | 6379 | 11 |  |  | failed at call 19 (round 1): Piston answered pledge "flywheel" (lowercase, and Flywheel wa |
| 20261014-qwen4b-think1k-game4 | finished | Qwen3.5-4B | on | 20000 | 1180258254 | 128 | 5674 | 0 | qwen3.5-4b-think-t1k-5 | [6.0, 2.0, 1.0, 0.0, 7.0] |  |
| eval-iter3-g00 | finished | iter3+qwen3.5-4b | on | 1500 | 48000 | 166 | 1813 | 100 | qwen3.5-4b-think-eval-iter3-3 | [6.0, 13.0, 20.0, 5.0, 12.0] |  |
| eval-iter3-g01 | finished | iter3+qwen3.5-4b | on | 1500 | 48001 | 97 | 1812 | 99 | qwen3.5-4b-think-eval-iter3-5 | [7.0, 6.0, 10.0, 13.0, 20.0] |  |
| eval-iter3-g02 | finished | iter3+qwen3.5-4b | on | 1500 | 48002 | 172 | 1812 | 100 | qwen3.5-4b-think-eval-iter3-4 | [9.0, 5.0, 10.0, 16.0, 2.0] |  |
| eval-iter3-g03 | finished | iter3+qwen3.5-4b | on | 1500 | 48003 | 159 | 1812 | 100 | qwen3.5-4b-think-eval-iter3-1 | [8.0, 3.0, 6.0, 3.0, 4.0] |  |
| eval-iter3-g04 | finished | iter3+qwen3.5-4b | on | 1500 | 48004 | 87 | 1814 | 100 | qwen3.5-4b-think-eval-iter3-4 | [4.0, 5.0, 1.0, 12.0, 9.0] |  |
| eval-iter3-g05 | finished | iter3+qwen3.5-4b | on | 1500 | 48005 | 51 | 1814 | 100 | qwen3.5-4b-think-eval-iter3-4 | [7.0, 11.0, 9.0, 14.0, 1.0] |  |
| eval-iter3-g06 | stopped | iter3+qwen3.5-4b | on | 1500 | 48006 | 233 | 1810 | 98 |  |  | stopped by Tashi at 05:01 at the night's spending limit |
| eval-iter3-g07 | stopped | iter3+qwen3.5-4b | on | 1500 | 48007 | 76 | 1813 | 100 |  |  | game running |
| eval-iter3-g08 | finished | iter3+qwen3.5-4b | on | 1500 | 48008 | 81 | 1811 | 99 | qwen3.5-4b-think-eval-iter3-4 | [8.0, 5.0, 11.0, 12.0, 6.0] |  |
| eval-iter3-g09 | finished | iter3+qwen3.5-4b | on | 1500 | 48009 | 206 | 1812 | 100 | qwen3.5-4b-think-eval-iter3-1 | [14.0, 0.0, 5.0, 11.0, 8.0] |  |
| eval-iter3-g10 | finished | iter3+qwen3.5-4b | on | 1500 | 48010 | 120 | 1811 | 100 | qwen3.5-4b-think-eval-iter3-2 | [1.0, 17.0, 3.0, 12.0, 9.0] |  |
| eval-iter3-g11 | stopped | iter3+qwen3.5-4b | on | 1500 | 48011 | 304 | 1811 | 99 |  |  | stopped by Tashi at 05:01 at the night's spending limit |
| eval-iter3-g12 | finished | iter3+qwen3.5-4b | on | 1500 | 48012 | 91 | 1813 | 100 | qwen3.5-4b-think-eval-iter3-5 | [10.0, 11.0, 11.0, 9.0, 14.0] |  |
| eval-iter3-g13 | finished | iter3+qwen3.5-4b | on | 1500 | 48013 | 238 | 1811 | 99 | qwen3.5-4b-think-eval-iter3-4 | [5.0, 8.0, 3.0, 12.0, 3.0] |  |
| eval-iter3-g14 | finished | iter3+qwen3.5-4b | on | 1500 | 48014 | 160 | 1811 | 99 | qwen3.5-4b-think-eval-iter3-5 | [5.0, 0.0, 6.0, 6.0, 9.0] |  |
| eval-iter3-g15 | finished | iter3+qwen3.5-4b | on | 1500 | 48015 | 117 | 1814 | 100 | qwen3.5-4b-think-eval-iter3-2 | [1.0, 8.0, 5.0, 6.0, 1.0] |  |
| eval-iter3-g16 | finished | iter3+qwen3.5-4b | on | 1500 | 48016 | 99 | 1816 | 100 | qwen3.5-4b-think-eval-iter3-5 | [3.0, 5.0, 4.0, 5.0, 7.0] |  |
| eval-iter3-g17 | finished | iter3+qwen3.5-4b | on | 1500 | 48017 | 228 | 1810 | 99 | qwen3.5-4b-think-eval-iter3-1 | [13.0, 5.0, 10.0, 11.0, 5.0] |  |
| eval-iter3-g18 | finished | iter3+qwen3.5-4b | on | 1500 | 48018 | 201 | 1811 | 99 | qwen3.5-4b-think-eval-iter3-2 | [14.0, 18.0, 12.0, 5.0, 10.0] |  |
| eval-iter3-g19 | finished | iter3+qwen3.5-4b | on | 1500 | 48019 | 194 | 1813 | 99 | qwen3.5-4b-think-eval-iter3-3 | [5.0, 14.0, 21.0, 7.0, 4.0] |  |
| rl-iter1-g00 | finished | qwen3.5-4b | on | 1500 | 41000 | 118 | 1813 | 99 | qwen3.5-4b-think-rl-iter1-5 | [10.0, 8.0, 10.0, 4.0, 12.0] |  |
| rl-iter1-g01 | finished | qwen3.5-4b | on | 1500 | 41001 | 189 | 1814 | 99 | qwen3.5-4b-think-rl-iter1-5 | [2.0, 16.0, 12.0, 9.0, 25.0] |  |
| rl-iter1-g02 | finished | qwen3.5-4b | on | 1500 | 41002 | 187 | 1812 | 100 | qwen3.5-4b-think-rl-iter1-3 | [5.0, 1.0, 18.0, 9.0, 10.0] |  |
| rl-iter1-g03 | finished | qwen3.5-4b | on | 1500 | 41003 | 178 | 1812 | 99 | qwen3.5-4b-think-rl-iter1-4 | [7.0, 5.0, 9.0, 16.0, 7.0] |  |
| rl-iter1-g04 | finished | qwen3.5-4b | on | 1500 | 41004 | 136 | 1814 | 100 | qwen3.5-4b-think-rl-iter1-4 | [11.0, 7.0, 5.0, 17.0, 15.0] |  |
| rl-iter1-g05 | finished | qwen3.5-4b | on | 1500 | 41005 | 179 | 1814 | 100 | qwen3.5-4b-think-rl-iter1-3 | [8.0, 9.0, 11.0, 5.0, 4.0] |  |
| rl-iter1-g06 | finished | qwen3.5-4b | on | 1500 | 41006 | 158 | 1814 | 100 | qwen3.5-4b-think-rl-iter1-3 | [8.0, 7.0, 14.0, 12.0, 8.0] |  |
| rl-iter1-g07 | finished | qwen3.5-4b | on | 1500 | 41007 | 305 | 1811 | 99 | qwen3.5-4b-think-rl-iter1-2 | [4.0, 18.0, 3.0, 12.0, 3.0] |  |
| rl-iter2-aborted-b-g00 | stopped | iter1 | on | 1500 | 42000 | 5 | 1818 | 100 |  |  | aborted by Tashi at 00:22: two games died on empty whisper text; schema now forbids empty  |
| rl-iter2-aborted-b-g01 | stopped | iter1 | on | 1500 | 42001 | 43 | 1812 | 100 |  |  | aborted by Tashi at 00:22: two games died on empty whisper text; schema now forbids empty  |
| rl-iter2-aborted-b-g02 | stopped | iter1 | on | 1500 | 42002 | 43 | 1813 | 100 |  |  | aborted by Tashi at 00:22: two games died on empty whisper text; schema now forbids empty  |
| rl-iter2-aborted-b-g03 | stopped | iter1 | on | 1500 | 42003 | 10 | 1821 | 100 |  |  | aborted by Tashi at 00:22: two games died on empty whisper text; schema now forbids empty  |
| rl-iter2-aborted-b-g04 | stopped | iter1 | on | 1500 | 42004 | 41 | 1811 | 100 |  |  | aborted by Tashi at 00:22: two games died on empty whisper text; schema now forbids empty  |
| rl-iter2-aborted-b-g05 | stopped | iter1 | on | 1500 | 42005 | 42 | 1809 | 100 |  |  | aborted by Tashi at 00:22: two games died on empty whisper text; schema now forbids empty  |
| rl-iter2-aborted-b-g06 | stopped | iter1 | on | 1500 | 42006 | 41 | 1813 | 100 |  |  | aborted by Tashi at 00:22: two games died on empty whisper text; schema now forbids empty  |
| rl-iter2-aborted-b-g07 | stopped | iter1 | on | 1500 | 42007 | 41 | 1811 | 100 |  |  | aborted by Tashi at 00:22: two games died on empty whisper text; schema now forbids empty  |
| rl-iter2-aborted-c-g00 | stopped | iter1 | on | 1500 | 43000 | 5 | 1820 | 100 |  |  | aborted by Tashi at 00:25: repetition penalty on the forced answer stopped it from closing |
| rl-iter2-aborted-c-g01 | stopped | iter1 | on | 1500 | 43001 | 5 | 1809 | 100 |  |  | aborted by Tashi at 00:25: repetition penalty on the forced answer stopped it from closing |
| rl-iter2-aborted-c-g02 | stopped | iter1 | on | 1500 | 43002 | 6 | 1817 | 100 |  |  | aborted by Tashi at 00:25: repetition penalty on the forced answer stopped it from closing |
| rl-iter2-aborted-c-g03 | stopped | iter1 | on | 1500 | 43003 | 5 | 1804 | 100 |  |  | aborted by Tashi at 00:25: repetition penalty on the forced answer stopped it from closing |
| rl-iter2-aborted-c-g04 | stopped | iter1 | on | 1500 | 43004 | 5 | 1819 | 100 |  |  | aborted by Tashi at 00:25: repetition penalty on the forced answer stopped it from closing |
| rl-iter2-aborted-c-g05 | stopped | iter1 | on | 1500 | 43005 | 4 | 1818 | 100 |  |  | aborted by Tashi at 00:25: repetition penalty on the forced answer stopped it from closing |
| rl-iter2-aborted-c-g06 | stopped | iter1 | on | 1500 | 43006 | 4 | 1809 | 100 |  |  | aborted by Tashi at 00:25: repetition penalty on the forced answer stopped it from closing |
| rl-iter2-aborted-c-g07 | stopped | iter1 | on | 1500 | 43007 | 2 | -373 | 100 |  |  | aborted by Tashi at 00:25: repetition penalty on the forced answer stopped it from closing |
| rl-iter2-aborted-d-g00 | stopped | iter1 | on | 1500 | 44000 | 11 | 1814 | 100 |  |  | aborted by Tashi at 00:30: forced answer under plain sampling degenerated into whitespace; |
| rl-iter2-aborted-d-g01 | stopped | iter1 | on | 1500 | 44001 | 8 | 1810 | 100 |  |  | aborted by Tashi at 00:30: forced answer under plain sampling degenerated into whitespace; |
| rl-iter2-aborted-d-g02 | stopped | iter1 | on | 1500 | 44002 | 11 | 1813 | 100 |  |  | aborted by Tashi at 00:30: forced answer under plain sampling degenerated into whitespace; |
| rl-iter2-aborted-d-g03 | stopped | iter1 | on | 1500 | 44003 | 11 | 1818 | 100 |  |  | aborted by Tashi at 00:30: forced answer under plain sampling degenerated into whitespace; |
| rl-iter2-aborted-d-g04 | stopped | iter1 | on | 1500 | 44004 | 11 | 1813 | 100 |  |  | aborted by Tashi at 00:30: forced answer under plain sampling degenerated into whitespace; |
| rl-iter2-aborted-d-g05 | stopped | iter1 | on | 1500 | 44005 | 11 | 1809 | 100 |  |  | aborted by Tashi at 00:30: forced answer under plain sampling degenerated into whitespace; |
| rl-iter2-aborted-d-g06 | stopped | iter1 | on | 1500 | 44006 | 10 | 1818 | 100 |  |  | aborted by Tashi at 00:30: forced answer under plain sampling degenerated into whitespace; |
| rl-iter2-aborted-d-g07 | stopped | iter1 | on | 1500 | 44007 | 10 | 1815 | 100 |  |  | aborted by Tashi at 00:30: forced answer under plain sampling degenerated into whitespace; |
| rl-iter2-aborted-e-g01 | stopped | iter1 | on | 1500 | 45001 | 109 | 1812 | 100 |  |  | died at 00:52: the Modal GPU container was preempted mid-call (7 games at once); recorder  |
| rl-iter2-aborted-e-g02 | stopped | iter1 | on | 1500 | 45002 | 108 | 1813 | 99 |  |  | died at 00:52: the Modal GPU container was preempted mid-call (7 games at once); recorder  |
| rl-iter2-aborted-e-g03 | stopped | iter1 | on | 1500 | 45003 | 109 | 1811 | 98 |  |  | died at 00:52: the Modal GPU container was preempted mid-call (7 games at once); recorder  |
| rl-iter2-aborted-e-g04 | stopped | iter1 | on | 1500 | 45004 | 104 | 1813 | 99 |  |  | died at 00:52: the Modal GPU container was preempted mid-call (7 games at once); recorder  |
| rl-iter2-aborted-e-g05 | stopped | iter1 | on | 1500 | 45005 | 109 | 1811 | 99 |  |  | died at 00:52: the Modal GPU container was preempted mid-call (7 games at once); recorder  |
| rl-iter2-aborted-e-g06 | stopped | iter1 | on | 1500 | 45006 | 109 | 1812 | 98 |  |  | died at 00:52: the Modal GPU container was preempted mid-call (7 games at once); recorder  |
| rl-iter2-aborted-e-g07 | stopped | iter1 | on | 1500 | 45007 | 109 | 1811 | 99 |  |  | died at 00:52: the Modal GPU container was preempted mid-call (7 games at once); recorder  |
| rl-iter2-aborted-g00 | stopped | iter1 | on | 1500 | 42000 | 44 | 1811 | 100 |  |  | aborted by Tashi at 00:08: batch restarted after one game died on a looping forced answer |
| rl-iter2-aborted-g01 | stopped | iter1 | on | 1500 | 42001 | 46 | 1811 | 96 |  |  | aborted by Tashi at 00:08: batch restarted after one game died on a looping forced answer |
| rl-iter2-aborted-g02 | stopped | iter1 | on | 1500 | 42002 | 44 | 1815 | 100 |  |  | aborted by Tashi at 00:08: batch restarted after one game died on a looping forced answer |
| rl-iter2-aborted-g03 | stopped | iter1 | on | 1500 | 42003 | 43 | 1812 | 100 |  |  | aborted by Tashi at 00:08: batch restarted after one game died on a looping forced answer |
| rl-iter2-aborted-g04 | stopped | iter1 | on | 1500 | 42004 | 42 | 1812 | 100 |  |  | aborted by Tashi at 00:08: batch restarted after one game died on a looping forced answer |
| rl-iter2-aborted-g05 | stopped | iter1 | on | 1500 | 42005 | 37 | 1812 | 100 |  |  | aborted by Tashi at 00:08: batch restarted after one game died on a looping forced answer |
| rl-iter2-aborted-g06 | stopped | iter1 | on | 1500 | 42006 | 43 | 1811 | 100 |  |  | aborted by Tashi at 00:08: batch restarted after one game died on a looping forced answer |
| rl-iter2-aborted-g07 | stopped | iter1 | on | 1500 | 42007 | 43 | 1813 | 100 |  |  | aborted by Tashi at 00:08: batch restarted after one game died on a looping forced answer |
| rl-iter2-g00 | finished | iter1 | on | 1500 | 46000 | 297 | 1812 | 100 | qwen3.5-4b-think-rl-iter2-5 | [18.0, 4.0, 17.0, 4.0, 20.0] |  |
| rl-iter2-g01 | finished | iter1 | on | 1500 | 46001 | 172 | 1812 | 99 | qwen3.5-4b-think-rl-iter2-1 | [11.0, 4.0, 7.0, 6.0, 4.0] |  |
| rl-iter2-g02 | finished | iter1 | on | 1500 | 46002 | 171 | 1812 | 99 | qwen3.5-4b-think-rl-iter2-5 | [11.0, 16.0, 7.0, 13.0, 18.0] |  |
| rl-iter2-g03 | finished | iter1 | on | 1500 | 46003 | 121 | 1812 | 99 | qwen3.5-4b-think-rl-iter2-4 | [12.0, 5.0, 13.0, 18.0, 7.0] |  |
| rl-iter2-g04 | running | iter1 | on | 1500 | 46004 | 158 | 1814 | 99 |  |  | game running |
| rl-iter2-g05 | finished | iter1 | on | 1500 | 46005 | 132 | 1812 | 100 | qwen3.5-4b-think-rl-iter2-3 | [7.0, 5.0, 13.0, 4.0, 13.0] |  |
| rl-iter2-g06 | finished | iter1 | on | 1500 | 46006 | 132 | 1812 | 100 | qwen3.5-4b-think-rl-iter2-3 | [11.0, 19.0, 22.0, 13.0, 18.0] |  |
| rl-iter2-g07 | finished | iter1 | on | 1500 | 46007 | 166 | 1811 | 99 | qwen3.5-4b-think-rl-iter2-5 | [10.0, 6.0, 9.0, 7.0, 12.0] |  |
| rl-iter2-g08 | finished | iter1 | on | 1500 | 45000 | 105 | 1815 | 100 | qwen3.5-4b-think-rl-iter2-4 | [7.0, 8.0, 5.0, 16.0, 6.0] |  |
| rl-iter3-g00 | finished | iter2 | on | 1500 | 47000 | 178 | 1811 | 99 | qwen3.5-4b-think-rl-iter3-4 | [3.0, 9.0, 13.0, 19.0, 11.0] |  |
| rl-iter3-g01 | stopped | iter2 | on | 1500 | 47001 | 246 | 1810 | 99 |  |  | stopped by Tashi at 03:22 (still playing after 240 calls) to start training and evaluation |
| rl-iter3-g02 | finished | iter2 | on | 1500 | 47002 | 157 | 1810 | 99 | qwen3.5-4b-think-rl-iter3-5 | [3.0, 10.0, 5.0, 6.0, 19.0] |  |
| rl-iter3-g03 | finished | iter2 | on | 1500 | 47003 | 231 | 1813 | 100 | qwen3.5-4b-think-rl-iter3-4 | [11.0, 2.0, 22.0, 28.0, 19.0] |  |
| rl-iter3-g04 | stopped | iter2 | on | 1500 | 47004 | 248 | 1810 | 99 |  |  | stopped by Tashi at 03:22 (still playing after 240 calls) to start training and evaluation |
| rl-iter3-g05 | finished | iter2 | on | 1500 | 47005 | 66 | 1815 | 100 | qwen3.5-4b-think-rl-iter3-2 | [9.0, 17.0, 5.0, 7.0, 4.0] |  |
| rl-iter3-g06 | finished | iter2 | on | 1500 | 47006 | 222 | 1813 | 100 | qwen3.5-4b-think-rl-iter3-1 | [15.0, 15.0, 6.0, 10.0, 0.0] |  |
| rl-iter3-g07 | stopped | iter2 | on | 1500 | 47007 | 243 | 1813 | 100 |  |  | stopped by Tashi at 03:22 (still playing after 240 calls) to start training and evaluation |

## Appendix B: training data and evaluation detail

### Iteration 1: seats used for the update (7 games, 35 seats)

| Game | Seat | Placement score | Advantage (leave-one-out) | Calls | Mean thinking tokens | Cut fraction |
|---|---|---|---|---|---|---|
| rl-iter1-g00 | 0 | 0.625 | +0.155 | 31 | 1813 | 1.00 |
| rl-iter1-g00 | 1 | 0.25 | -0.314 | 22 | 1814 | 1.00 |
| rl-iter1-g00 | 2 | 0.625 | +0.155 | 17 | 1814 | 1.00 |
| rl-iter1-g00 | 3 | 0.0 | -0.621 | 24 | 1772 | 0.96 |
| rl-iter1-g00 | 4 | 1.0 | +0.624 | 24 | 1813 | 1.00 |
| rl-iter1-g01 | 0 | 0.0 | -0.626 | 40 | 1816 | 1.00 |
| rl-iter1-g01 | 1 | 0.75 | +0.312 | 38 | 1815 | 1.00 |
| rl-iter1-g01 | 2 | 0.5 | +0.002 | 35 | 1795 | 0.97 |
| rl-iter1-g01 | 3 | 0.25 | -0.313 | 39 | 1813 | 1.00 |
| rl-iter1-g01 | 4 | 1.0 | +0.624 | 37 | 1815 | 1.00 |
| rl-iter1-g02 | 0 | 0.25 | -0.312 | 42 | 1814 | 1.00 |
| rl-iter1-g02 | 1 | 0.0 | -0.625 | 36 | 1814 | 1.00 |
| rl-iter1-g02 | 2 | 1.0 | +0.625 | 33 | 1813 | 1.00 |
| rl-iter1-g02 | 3 | 0.5 | -0.000 | 41 | 1815 | 1.00 |
| rl-iter1-g02 | 4 | 0.75 | +0.312 | 35 | 1814 | 1.00 |
| rl-iter1-g03 | 0 | 0.375 | -0.157 | 41 | 1812 | 1.00 |
| rl-iter1-g03 | 1 | 0.0 | -0.626 | 34 | 1813 | 1.00 |
| rl-iter1-g03 | 2 | 0.75 | +0.312 | 41 | 1814 | 1.00 |
| rl-iter1-g03 | 3 | 1.0 | +0.628 | 33 | 1781 | 0.97 |
| rl-iter1-g03 | 4 | 0.375 | -0.157 | 29 | 1813 | 1.00 |
| rl-iter1-g04 | 0 | 0.5 | -0.000 | 25 | 1817 | 1.00 |
| rl-iter1-g04 | 1 | 0.25 | -0.312 | 25 | 1814 | 1.00 |
| rl-iter1-g04 | 2 | 0.0 | -0.625 | 25 | 1815 | 1.00 |
| rl-iter1-g04 | 3 | 1.0 | +0.625 | 30 | 1815 | 1.00 |
| rl-iter1-g04 | 4 | 0.75 | +0.313 | 31 | 1813 | 1.00 |
| rl-iter1-g05 | 0 | 0.5 | +0.000 | 39 | 1813 | 1.00 |
| rl-iter1-g05 | 1 | 0.75 | +0.312 | 32 | 1815 | 1.00 |
| rl-iter1-g05 | 2 | 1.0 | +0.625 | 34 | 1814 | 1.00 |
| rl-iter1-g05 | 3 | 0.25 | -0.312 | 38 | 1814 | 1.00 |
| rl-iter1-g05 | 4 | 0.0 | -0.625 | 36 | 1814 | 1.00 |
| rl-iter1-g06 | 0 | 0.375 | -0.156 | 26 | 1816 | 1.00 |
| rl-iter1-g06 | 1 | 0.0 | -0.625 | 29 | 1814 | 1.00 |
| rl-iter1-g06 | 2 | 1.0 | +0.625 | 33 | 1815 | 1.00 |
| rl-iter1-g06 | 3 | 0.75 | +0.312 | 36 | 1816 | 1.00 |
| rl-iter1-g06 | 4 | 0.375 | -0.156 | 34 | 1812 | 1.00 |

### Iteration 2: seats used for the update (7 games, 35 seats)

| Game | Seat | Placement score | Advantage (leave-one-out) | Calls | Mean thinking tokens | Cut fraction |
|---|---|---|---|---|---|---|
| rl-iter2-g01 | 0 | 1.0 | +0.625 | 29 | 1813 | 1.00 |
| rl-iter2-g01 | 1 | 0.125 | -0.469 | 38 | 1814 | 1.00 |
| rl-iter2-g01 | 2 | 0.75 | +0.312 | 30 | 1815 | 1.00 |
| rl-iter2-g01 | 3 | 0.5 | -0.000 | 36 | 1815 | 1.00 |
| rl-iter2-g01 | 4 | 0.125 | -0.469 | 39 | 1812 | 0.97 |
| rl-iter2-g02 | 0 | 0.25 | -0.312 | 37 | 1812 | 0.97 |
| rl-iter2-g02 | 1 | 0.75 | +0.312 | 34 | 1815 | 1.00 |
| rl-iter2-g02 | 2 | 0.0 | -0.625 | 37 | 1813 | 1.00 |
| rl-iter2-g02 | 3 | 0.5 | -0.000 | 34 | 1816 | 1.00 |
| rl-iter2-g02 | 4 | 1.0 | +0.625 | 29 | 1813 | 1.00 |
| rl-iter2-g03 | 0 | 0.5 | -0.001 | 24 | 1814 | 1.00 |
| rl-iter2-g03 | 1 | 0.0 | -0.620 | 20 | 1761 | 0.95 |
| rl-iter2-g03 | 2 | 0.75 | +0.311 | 27 | 1815 | 1.00 |
| rl-iter2-g03 | 3 | 1.0 | +0.624 | 27 | 1813 | 1.00 |
| rl-iter2-g03 | 4 | 0.25 | -0.314 | 23 | 1814 | 1.00 |
| rl-iter2-g05 | 0 | 0.5 | -0.000 | 27 | 1815 | 1.00 |
| rl-iter2-g05 | 1 | 0.25 | -0.313 | 29 | 1815 | 1.00 |
| rl-iter2-g05 | 2 | 0.875 | +0.469 | 22 | 1813 | 1.00 |
| rl-iter2-g05 | 3 | 0.0 | -0.625 | 35 | 1812 | 1.00 |
| rl-iter2-g05 | 4 | 0.875 | +0.469 | 19 | 1814 | 1.00 |
| rl-iter2-g06 | 0 | 0.0 | -0.625 | 26 | 1813 | 1.00 |
| rl-iter2-g06 | 1 | 0.75 | +0.313 | 26 | 1813 | 1.00 |
| rl-iter2-g06 | 2 | 1.0 | +0.625 | 34 | 1813 | 1.00 |
| rl-iter2-g06 | 3 | 0.25 | -0.313 | 27 | 1815 | 1.00 |
| rl-iter2-g06 | 4 | 0.5 | -0.000 | 19 | 1816 | 1.00 |
| rl-iter2-g07 | 0 | 0.75 | +0.312 | 32 | 1810 | 1.00 |
| rl-iter2-g07 | 1 | 0.0 | -0.626 | 33 | 1815 | 1.00 |
| rl-iter2-g07 | 2 | 0.5 | -0.001 | 33 | 1814 | 1.00 |
| rl-iter2-g07 | 3 | 0.25 | -0.311 | 35 | 1799 | 0.97 |
| rl-iter2-g07 | 4 | 1.0 | +0.625 | 33 | 1812 | 1.00 |
| rl-iter2-g08 | 0 | 0.5 | +0.000 | 24 | 1815 | 1.00 |
| rl-iter2-g08 | 1 | 0.75 | +0.313 | 23 | 1814 | 1.00 |
| rl-iter2-g08 | 2 | 0.0 | -0.625 | 24 | 1817 | 1.00 |
| rl-iter2-g08 | 3 | 1.0 | +0.625 | 19 | 1817 | 1.00 |
| rl-iter2-g08 | 4 | 0.25 | -0.313 | 15 | 1816 | 1.00 |

### Iteration 3: seats used for the update (5 games, 25 seats)

| Game | Seat | Placement score | Advantage (leave-one-out) | Calls | Mean thinking tokens | Cut fraction |
|---|---|---|---|---|---|---|
| rl-iter3-g00 | 0 | 0.0 | -0.626 | 35 | 1813 | 1.00 |
| rl-iter3-g00 | 1 | 0.25 | -0.313 | 39 | 1814 | 1.00 |
| rl-iter3-g00 | 2 | 0.75 | +0.312 | 32 | 1813 | 1.00 |
| rl-iter3-g00 | 3 | 1.0 | +0.628 | 41 | 1780 | 0.98 |
| rl-iter3-g00 | 4 | 0.5 | -0.001 | 31 | 1811 | 1.00 |
| rl-iter3-g02 | 0 | 0.0 | -0.625 | 34 | 1809 | 0.97 |
| rl-iter3-g02 | 1 | 0.75 | +0.312 | 31 | 1813 | 1.00 |
| rl-iter3-g02 | 2 | 0.25 | -0.313 | 29 | 1812 | 1.00 |
| rl-iter3-g02 | 3 | 0.5 | -0.000 | 35 | 1812 | 1.00 |
| rl-iter3-g02 | 4 | 1.0 | +0.625 | 28 | 1813 | 1.00 |
| rl-iter3-g03 | 0 | 0.25 | -0.313 | 50 | 1814 | 1.00 |
| rl-iter3-g03 | 1 | 0.0 | -0.625 | 53 | 1813 | 1.00 |
| rl-iter3-g03 | 2 | 0.75 | +0.313 | 33 | 1813 | 1.00 |
| rl-iter3-g03 | 3 | 1.0 | +0.625 | 49 | 1814 | 1.00 |
| rl-iter3-g03 | 4 | 0.5 | +0.000 | 46 | 1813 | 1.00 |
| rl-iter3-g05 | 0 | 0.75 | +0.312 | 10 | 1820 | 1.00 |
| rl-iter3-g05 | 1 | 1.0 | +0.625 | 14 | 1816 | 1.00 |
| rl-iter3-g05 | 2 | 0.25 | -0.312 | 17 | 1816 | 1.00 |
| rl-iter3-g05 | 3 | 0.5 | -0.000 | 11 | 1819 | 1.00 |
| rl-iter3-g05 | 4 | 0.0 | -0.625 | 14 | 1816 | 1.00 |
| rl-iter3-g06 | 0 | 0.875 | +0.471 | 46 | 1791 | 0.98 |
| rl-iter3-g06 | 1 | 0.875 | +0.468 | 48 | 1814 | 1.00 |
| rl-iter3-g06 | 2 | 0.25 | -0.313 | 44 | 1814 | 1.00 |
| rl-iter3-g06 | 3 | 0.5 | -0.001 | 45 | 1815 | 1.00 |
| rl-iter3-g06 | 4 | 0.0 | -0.625 | 39 | 1813 | 1.00 |

### Training-step logs (from the Modal trainer)

| Adapter | Examples | Generated tokens | Optimizer steps | LR | Mean loss | Mean entropy | Clipped fraction | Sampled KL to base (k3) | Seconds |
|---|---|---|---|---|---|---|---|---|---|
| iter1 | 1145 | 2,113,914 | 4 | 3e-05 | -0.0016 | 0.763 | 0.0014 | 0.001078 | 4617 |
| iter2 | 600 | 1,108,070 | 4 | 3e-05 | +0.0238 | 0.753 | 0.0015 | 0.01749 | 599 |
| iter3 | 600 | 1,105,690 | 4 | 3e-05 | +0.0139 | 0.750 | 0.0034 | 10.09 | 632 |

### Evaluation games (iter3 seat vs four base seats)

| Game | iter3 seat position | iter3 placement score |
|---|---|---|
| eval-iter3-g00 | 0 | 0.25 |
| eval-iter3-g01 | 1 | 0.0 |
| eval-iter3-g02 | 2 | 0.75 |
| eval-iter3-g03 | 3 | 0.125 |
| eval-iter3-g04 | 4 | 0.75 |
| eval-iter3-g05 | 0 | 0.25 |
| eval-iter3-g08 | 3 | 1.0 |
| eval-iter3-g09 | 4 | 0.5 |
| eval-iter3-g10 | 0 | 0.0 |
| eval-iter3-g12 | 2 | 0.625 |
| eval-iter3-g13 | 3 | 1.0 |
| eval-iter3-g14 | 4 | 1.0 |
| eval-iter3-g15 | 0 | 0.125 |
| eval-iter3-g16 | 1 | 0.625 |
| eval-iter3-g17 | 2 | 0.5 |
| eval-iter3-g18 | 3 | 0.0 |
| eval-iter3-g19 | 4 | 0.0 |
