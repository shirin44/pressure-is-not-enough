# 02 — Demonstration seeding: single domain (door latch)

## What we tried
SFT-seeded the model on an unrelated toy domain (door latch, Locked/
Unlocked) to explicitly teach "track a hidden binary state via a
consistent substitute token," as a disclosed deviation from purely
emergent discovery, before attempting to transfer that skill elsewhere.

## Why we tried it
Direct RL alone (`01_direct_rl_coinflip`) produced zero verified
encodings within budget. Demonstration-seeding is the disclosed
alternative: teach the strategy first, hope it transfers.

## Key steps / iterations
1. **Issue**: unpredictable random target words collapsed to one repeated
   meaningless word. **Fix**: explicitly declare the code-to-state mapping
   in the prompt.
2. **Issue**: model tracked the encoding correctly but copied the *code
   word itself* into `<answer>` instead of decoding back to the physical
   state. **Fix**: added an explicit decode-back line before the answer.
3. **Issue**: ~92% accuracy but the model read the declared mapping
   backwards at the start of the trace (73% of remaining errors). **Fix**:
   added an explicit "initial code anchor" line immediately after the
   mapping declaration.
4. A checkpoint-130-to-140 explicit-transition diagnostic reduced pure
   transition errors (19→14) but reintroduced 25 consistent mapping swaps
   and dropped answer accuracy to 79%; **checkpoint 140 rejected**, not
   used downstream.
5. Zero-shot generalization test (Stage 3.5, single-domain) on an
   unrelated third domain (fan): **failed badly** — 0% global consistency,
   3% accuracy. A matched control on the *unadapted base model* scored
   equally poorly (3% adherence vs. checkpoint 130's 5%), ruling out "the
   latch SFT damaged a pre-existing capability" — the task is intrinsically
   hard zero-shot at this model scale from a single seeding domain.

## Final result
Checkpoint 130 accepted for its own task under a registered threshold
exception (95% target, not fully cleared): accuracy 92.5%, structure 100%,
tracking 90.5%, global consistency 89.0%, mapping adherence 89.5%,
decode-back 92.0%, nonliteral use 100%. **But zero cross-domain
generalization** — the core capability did not transfer to an unseen
domain.

## What this means for the next step
A single seeding domain teaches domain-specific surface behavior, not the
abstract strategy. Retrain from scratch across *multiple* simultaneous
training domains, holding a third domain out entirely for zero-shot
testing — see `03_demonstration_seeding_multi_domain`.

---
**Checkpoint-130 naming collision**: this experiment's final checkpoint is
`latch-seed-stage3-decode-back-sft-v3/trainer-output/checkpoint-130` on
Drive. `03_demonstration_seeding_multi_domain` *also* has a checkpoint
numbered 130 (`multidomain-stage3-v1/trainer-output/checkpoint-130`) —
same number, **completely different, unrelated training lineage**. Do not
confuse the two.
