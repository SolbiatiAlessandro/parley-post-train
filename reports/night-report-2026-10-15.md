# Overnight report: Parley post-training, night of 2026-10-14 → 15

*Draft, being filled in during the night. Tagged: (verified) = measured from the recordings; (my read) = my judgment.*

## Short version

- Game 4 (Qwen3.5-4B, thinking on, told "under 1000 tokens of thinking", not enforced): the instruction had no effect. Median thinking 5,674 tokens, range 434–12,878 (verified).
- RL loop built and running: self-play rollouts on a Modal H100, LoRA policy-gradient updates (algorithm from the literature review, below), evaluation planned as 1 trained seat vs 4 untouched seats.
- Results: see "Iterations" and "Evaluation" below.

## What ran

| Game / batch | Model and settings | Outcome |
|---|---|---|
| Game 1 | thinking off | finished, 122 calls, all legal |
| Game 2, 2b | thinking on, 3,000 / 2,000-token cutoff | failed / stopped: thinking never finished |
| Game 3 | thinking on, 8,000 cutoff, H100 | failed at call 19 on an illegal pledge name |
| Game 4 | thinking on, "under 1000 tokens" asked, 20,000 cap | finished, 128 calls; instruction ignored |
| RL iteration 1 rollouts | base model, 1,500 thinking budget enforced | 8/8 games finished (1,450 calls) |
| RL iteration 2 rollouts | adapter iter1 | see below |

## Algorithm (from reports/rl-literature-review.md)

Critic-free policy gradient. Group = the five seats of one game; advantage = a seat's placement score minus the mean of the other four seats (leave-one-out baseline, RLOO form); the same advantage on every token the seat generated (thinking and answer), token-level loss normalized by the batch's generated tokens, 4 clipped minibatch steps per iteration (clip 0.2 below, 0.28 above, old log-probs recomputed by the trainer), no KL term (KL to the base model logged on a 10% sample), LoRA rank 16 on all attention and MLP projections, learning rate 3e-5, AdamW. Thinking cut at 1,500 tokens with Qwen's own "limited time" sentence; the injected sentence is masked out of the loss. No length penalty: every rollout call hit the cutoff, so length carried no signal.

## Iterations

| Iteration | Policy that played | Games finished | Training calls | Optimizer steps | Train time | Entropy | KL to base (sampled) |
|---|---|---|---|---|---|---|---|
| 1 | base (iter0) | 8/8 (+0 lost) | 1,145 (all) | 4 | 77 min (slow kernels) | 0.76 | 0.001 |
| 2 | iter1 | 7 (after 4 aborted launches, see below) | 600 (capped) | 4 | 10 min (tilelang) | 0.75 | 0.017 |
| 3 | iter2 | 5 (3 long games stopped at 240 calls to start training) | 600 (capped) | 4 | 11 min | 0.75 | 10.1 (see note) |

Note on the KL column: it is the GRPO "k3" estimator averaged over sampled tokens on 10% of the calls. It is heavy-tailed: a single sampled token the base model finds very unlikely can dominate the mean, so the jump at iteration 3 is not by itself evidence of a broken policy (my read). The evaluation below is the real check.

## Evaluation

(planned: 20 games, one trained seat vs four base seats, rotating the seat; metric = the trained seat's mean placement score, 0.5 = no change, standard error about 0.08)

## What broke, and the fixes (verified)

1. Prompt-only thinking limits are ignored by this model (game 4).
2. Decoder-level name constraints (JSON schema built from each prompt's legal lists) stop illegal names from ending games.
3. Empty whisper text ended two games → schema requires 1–160-character strings.
4. A repetition penalty on the forced answer penalized the closing braces → removed.
5. Plain sampling (no top-p/top-k) made the grammar-constrained forced answer run off into whitespace → Qwen's sampling settings for the forced answer, plain sampling for the thinking.
6. Modal preempted the GPU container at 00:52 and seven mid-call games died → the recorder now retries a call across server restarts (up to 10 min).
7. Training without the fast linear-attention kernels took 77 min for one update → image now has flash-linear-attention plus tilelang (its Triton backward refuses to run on H100 without it); a 600-call update takes 10 min. causal-conv1d failed to build (slow fallback for that small part).
8. One game died when the model closed its thinking just before the budget and the JSON itself was cut off → that case now goes through the same "limited time" continuation.

## Cost

(filled in at the end from `modal billing report`)
