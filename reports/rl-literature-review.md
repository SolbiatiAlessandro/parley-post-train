# RL for Parley (Qwen3.5-4B, LoRA, 5-seat self-play): literature review and spec

2026-10-14. Tags: **verified** = in the abstract or paper text I fetched; **fairly sure** = secondary page or fetch summary; **guess** = my inference or arithmetic.

**Bottom line.** Critic-free REINFORCE, table-level leave-one-out baseline, token-level loss, one optimizer step per batch, no KL, LoRA rank 32 at 5e-5, plus a bounded per-call thinking-length term. Tonight's 200-640 trajectories are ~1% of SPIRAL's 400 x 128: expect shifts in format and length, not deep strategy (guess).

## 1. Policy-gradient variants
- **GRPO** (Shao et al. 2024, DeepSeekMath, arXiv 2402.03300; group-relative policy optimization, G samples per prompt, no critic): advantage (r - group mean)/group std on every token; loss averaged per sequence (1/|o|); clipped ratio; KL 0.04 in the loss; lr 1e-6, batch 1,024, one update per rollout set (verified).
- **Dr. GRPO** (Liu et al. 2025, 2503.20783): 1/|o| and 1/std bias toward long wrong answers and reweight questions; use r - mean and a constant normalizer. lr 1e-6, KL 0 (verified).
- **DAPO** (Yu et al. 2025, 2503.14476): clip-higher 0.2/0.28, drop all-right/all-wrong groups, token-level loss over the whole batch, overlong ramp over the last 4,096 of 20,480 tokens then -1, no KL (verified).
- **GSPO** (Zheng et al. 2025, Qwen team, 2507.18071): sequence-level ratio (pi/pi_old)^(1/|y|), clip 3e-4/4e-4, since per-token ratios add noise (verified); moot when the ratio is 1 (guess).
- **REINFORCE++** (Hu et al. 2025, 2501.03262): normalize over the whole batch; per-prompt std with groups of 4-8 can explode (verified).
- **RLOO** (Ahmadian et al. 2024, 2402.14740; REINFORCE leave-one-out): baseline = mean of the other k-1 rewards, completion as one action; beat PPO (Schulman et al. 2017, 1707.06347; learned critic, skipped tonight) and RAFT (reward-ranked fine-tuning, trains on best samples) at equal samples (verified).
- "Tricks or Traps" (Liu et al. 2025, 2508.08221): group mean plus batch std is most robust; token-level loss for base models, sequence-level for aligned (fairly sure). ScaleRL (Khatri et al. 2025, 2510.13786; a large RL-compute study): aggregation and normalization change efficiency, not the asymptote (verified). 2-GRPO keeps 97.6% of 16-GRPO's performance (2510.00977, verified).

## 2. Credit assignment
- **SPIRAL** (Liu et al. 2025, 2506.24119; self-play multi-turn RL, Qwen3-4B-Base, two-player zero-sum games): pure REINFORCE; advantage = return - moving-average baseline per (game, role), decay 0.95, same value on every token of that player's turns; terminal +-1; no length normalization; 128 samples/step, 400 steps, lr 1e-6, KL 0, 8,192 max tokens, latest-policy self-play. Without the role baseline, thinking collapsed within ~100 steps (verified).
- **MARSHAL** (Yuan et al. 2025, 2510.15414; multi-agent self-play, Qwen3-4B): subtract per-role mean of summed rewards; batch 128-384, lr 1e-6, KL 0.2 (fairly sure). With terminal-only reward that is the same advantage on every turn (guess).
- **RAGEN/StarPO** (Wang et al. 2025, 2504.20073; multi-turn agent RL): collapse shows as reward-std drop, entropy swings, gradient spikes; fixes: keep top-25% reward-variance prompts, no KL, clip-higher (verified).
- ArCHer (2402.19446), SWEET-RL (2503.15478), MT-GRPO (2505.11821), GiGPO (2505.10978) need a turn-level critic, per-turn rewards or repeated states; none exists here (guess).
- Zhang et al. (2610.04261): terminal win/loss alone does not reliably teach social deduction (verified). OMAR (Jiang et al. 2026, 2602.03109; hidden-role Werewolf self-play) needed SFT warm start against reward hacking (fairly sure).

## 3. Self-play
- SPIRAL: self-play held 50-52% win rate; a fixed Google Gemini opponent went 0 to 62.5% (exploited); random opponents collapsed (verified). MARSHAL: fixed opponents degraded (fairly sure).
- Sarkar et al. (2025, 2502.06060; Among Us-style, 4 crew + 1 imposter, PPO) co-trained the imposter and froze one agent (fairly sure).
- I found no published LLM-RL result on a 5-player free-for-all with identical copies. Nearest: non-LLM self-play in 7-player Diplomacy (a negotiation board game) (2110.02924) learns conventions incompatible with human-data agents (verified).
- Under self-play mean reward is fixed by construction, so it cannot show learning (guess). Over 5-8 small steps non-stationarity is minor (guess).

## 4. Length control
- **L1/LCPO** (Aggarwal & Welleck 2025, 2503.04697; length-controlled policy optimization): reward = correct - alpha|n_target - n|, alpha 3e-4; lr 1e-6 (verified).
- **Kimi k1.5** (Moonshot AI's reasoning model, 2501.12599): lambda = 0.5 - (len - min)/(max - min); reward lambda if correct, min(0, lambda) if wrong; added after warm-up because it slows early learning (verified).
- Arora & Zanette (2025, 2502.04463): correct x (1 - alpha x sigmoid(z-scored length)), alpha 0.05-0.4; ~50% fewer tokens, <5% accuracy loss (verified).
- ThinkPrune (2504.01296): zero reward above a cap, tightened in stages; length halved, -2% (verified). Fatemi et al. (2504.05185): negative-advantage samples lengthen outputs (verified).

## 5. LoRA for RL
- "LoRA Without Regret" (Schulman et al., Thinking Machines Lab 2025): rank 1 matched full fine-tuning in RL (~1 bit per episode); all layers incl. MLP; alpha 32 with 1/r scaling; best LR ~10x full fine-tuning, ~15x under ~100 steps (verified).
- Counter-evidence: in long-reasoning RL, rank 1 scored 40.5% vs 42.5% standard LoRA vs 44.9% full (2512.23165) (verified). Hence rank 32 (guess).

## 6. Small-sample stability
- Entropy: performance ~ -a e^H + b (Cui et al. 2025, 2505.22617, verified).
- KL is 0 in DAPO, Dr. GRPO, SPIRAL; 0.001 in DeepSeek-R1 (2501.12948, fairly sure).
- With 40-80 trajectories, each reward is one draw (SD ~0.35) shared by ~30 calls: mostly noise (guess).
- Qwen's recommended thinking sampling (T 1.0, top-p 0.95, top-k 20, presence penalty 1.5; model card, verified) differs from the distribution the trainer scores; train/inference mismatch destabilizes RL (2510.26788, 2512.01374; verified).

## 7. Tokens, masking, memory
- Loss on generated tokens only, thinking included (SPIRAL, verified); mask prompts, keep the end token, store sampled token IDs (guess).
- Qwen3.5-4B: 32 layers (3 Gated DeltaNet linear-attention : 1 attention), hidden 2560, vocab 248,320 (verified). Without flash-linear-attention and causal-conv1d kernels training is slow (fairly sure).
- Per 6k-token sequence with checkpointing (guess): weights ~9 GB, adapter+Adam <1 GB, saved activations ~1 GB, recompute ~2 GB; fp32 logits are 6 GB (x2-3 with gradient), so score completion tokens in 1k chunks: ~20 GB total, fitting beside vLLM (inference server) at a 0.4-0.5 memory share.
- 7-14M tokens per iteration is ~10-25 min of training (guess).

# Recommendation for tonight

- **Group**: the 5 seats of one game. A_i = s_i - mean(s_j, j != i), s = placement score (RLOO form, unbiased even with ties); without ties that is 1.25(s_i - 0.5). No std division (near-constant here). Same A_i on all ~30 calls of that seat. Add per-seat baselines only if seat means differ by over 0.1.
- **Length term** (per call, thinking tokens T): lambda = clip(0.5 - (T - T10)/(T90 - T10), -0.5, 0.5), percentiles within call type (shoot/pass vs reaction). If A_i >= 0 add c x lambda, else c x min(0, lambda) (Kimi form). c = 0.2 in iteration 1, then 0.4, i.e. +-0.2 against game advantages of 0.3-0.6; per token within 2x of LCPO (guess). Truncated or invalid-JSON calls get -0.5 and stay in the loss.
- **Loss**: -sum over completion tokens of A_call x logprob, divided by the batch's total completion tokens. Token-level because 1/|o| favors long negative samples; this overrides Tricks-or-Traps' sequence-level advice (guess).
- **Steps**: one optimizer step per iteration: ratio = 1, no clipping, no old log-probs. With 2+ steps, recompute old log-probs in the trainer and clip 0.2/0.28.
- **KL**: none. Log sampled-token KL to base (adapter disabled) on 10% of sequences; roll back above ~0.05 (guess).
- **LoRA**: rank 32, alpha 32, dropout 0, on q,k,v,o,gate,up,down. lr 5e-5 constant, AdamW (0.9, 0.95), no weight decay, grad-norm clip 1.0. B starts at zero and Adam moves each weight ~lr per step, so 8 steps at 1e-5 may show nothing (guess). Halve if KL or length moves fast.
- **Batch**: micro-batch 1 sequence, gradient checkpointing, accumulate all; max prompt 4,096, max new tokens 6,144.
- **Rollouts**: T 1.0, top-p 1.0, no presence penalty, latest adapter on all five seats. If truncation exceeds 15%, restore Qwen's settings (guess).
- **Log**: completion entropy, thinking length (median, p90) by call type, truncation and invalid-JSON rates, shoot rate, pre-clip grad norm, per-seat mean score, sampled KL. Collapse flags: entropy under half of iteration 1, length under 30% of start without eval gain, grad-norm spike over 5x, rising invalid JSON.
- **Eval**: 20 games, one trained seat vs four frozen base seats, rotating (4 games per seat). If placements are uniform, a seat score has SD 0.354: standard error 0.08, 95% interval +-0.16, and a real +0.1 needs ~100 games (guess). Twenty games catch collapse or a gain near 0.22, not small gains. Two trained vs three base gives ~1.5x effective games (guess). Run base vs base once for noise.
- **Cheapest baseline**: rejection-sampling fine-tuning (RAFT). From the same games keep calls of seats scoring >= 0.75, and for length the shorter-thinking half (Kimi's "shortest rejection sampling"); SFT one epoch, same LoRA, ~2e-4. Xiong et al. (2504.11343) found RAFT++ 56.1% vs GRPO 56.3% (verified). Add a length-term-only control. RL beats RAFT only if it wins the same eval at equal games sampled.

## Sources (fetched; abs = abstract page, html = full text)
https://arxiv.org/abs/2506.24119
https://arxiv.org/html/2506.24119
https://arxiv.org/abs/2503.20783
https://arxiv.org/html/2503.20783
https://arxiv.org/abs/2503.14476
https://arxiv.org/html/2503.14476
https://arxiv.org/abs/2507.18071
https://arxiv.org/html/2507.18071
https://arxiv.org/abs/2501.03262
https://arxiv.org/html/2501.03262
https://arxiv.org/abs/2402.14740
https://arxiv.org/html/2402.14740
https://arxiv.org/abs/2402.03300
https://arxiv.org/html/2402.03300v3
https://arxiv.org/pdf/2402.03300 (unparseable, not used)
https://arxiv.org/html/2501.12948
https://arxiv.org/abs/1707.06347
https://arxiv.org/html/2504.20073
https://arxiv.org/abs/2402.19446
https://arxiv.org/abs/2505.11821
https://arxiv.org/html/2505.11821
https://arxiv.org/abs/2503.15478
https://arxiv.org/abs/2505.10978
https://arxiv.org/html/2505.10978
https://arxiv.org/abs/2502.06060
https://arxiv.org/html/2502.06060
https://arxiv.org/abs/2510.15414
https://arxiv.org/html/2510.15414
https://arxiv.org/abs/2602.03109
https://arxiv.org/html/2602.03109
https://arxiv.org/abs/2610.04261
https://arxiv.org/html/2610.04261
https://arxiv.org/abs/2606.23348
https://arxiv.org/abs/2110.02924
https://arxiv.org/abs/2310.18940
https://arxiv.org/abs/2602.05183
https://arxiv.org/abs/2503.04697
https://arxiv.org/html/2503.04697
https://arxiv.org/abs/2504.01296
https://arxiv.org/html/2501.12599
https://arxiv.org/abs/2502.04463
https://arxiv.org/html/2502.04463
https://arxiv.org/abs/2504.05185
https://arxiv.org/abs/2503.16419
https://arxiv.org/abs/2603.17449
https://thinkingmachines.ai/blog/lora/
https://huggingface.co/docs/trl/main/en/lora_without_regret
https://kalomaze.bearblog.dev/rl-lora-ddd/
https://arxiv.org/abs/2405.09673
https://arxiv.org/abs/2504.20834
https://arxiv.org/abs/2504.15777
https://arxiv.org/html/2512.23165v1
https://arxiv.org/abs/2505.22617
https://arxiv.org/abs/2504.11343
https://arxiv.org/html/2504.11343
https://arxiv.org/abs/2508.08221
https://arxiv.org/html/2508.08221
https://arxiv.org/abs/2510.13786
https://arxiv.org/abs/2512.01374
https://arxiv.org/abs/2510.00977
https://arxiv.org/abs/2510.26788
https://huggingface.co/Qwen/Qwen3.5-4B
https://unsloth.ai/docs/models/qwen3.5/fine-tune
https://swift.readthedocs.io/en/v4.4/_sources/BestPractices/Qwen3_5-Best-Practice.md.txt
Fetch failed (HTTP 429, not used): https://www.themoonlight.io/review/evaluating-parameter-efficient-methods-for-rlvr
