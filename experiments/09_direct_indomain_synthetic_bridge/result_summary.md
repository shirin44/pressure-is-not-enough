# Stage 9 interim result — Step 13 coded condition

## Official coded-condition result (partial diagnosis)

Step 13's coded condition is complete and is a clean negative on coded behavior
specifically; Stage 9 is not yet fully closed. The full bridge run continued the
milestone-12 Step-0 adapter for 150 optimizer steps at the fixed 10% coded
auxiliary-CE condition and delivered exactly 120/120 scheduled teacher-forced
Nib/Nomo examples. All 39 evaluations completed and no hard or soft breaker
fired.

The evaluations generated 819 clean held-out completions (21 per milestone)
and 624 bank-scenario completions (16 per milestone). All 1,443 were literal
(0/1,443 non-literal).
There were zero structural-nonliteral candidates, zero Nib/Nomo appearances,
zero Yelt/Yark appearances, and zero exact echoes of an injected completion.
Thus there was no memorization, partial coded transfer, clean-scenario coded
generalization, or held-out-pair invention.

This is the project's most direct test of coded behavior. Earlier stages tested
whether a code would arise spontaneously; this run supplied a demonstrated, verified
code directly through teacher-forced trajectories. The demonstrated code still
did not appear in sampled reasoning. That result does not diagnose whether the
auxiliary CE channel was strong enough to teach any behavior at all.

## Auxiliary-mechanism diagnosis and control clarification

Step 10 constructed a 16-example coded bank and a matched 16-example literal-
control bank as alternative conditions. The Step 13 implementation calls
`select_synthetic_examples(CODED_TRAJECTORIES, ...)`; it did **not** mix coded
and literal-control examples in this run. Consequently, the evidence cannot
show whether the auxiliary CE mechanism functions as a supervised-learning
channel at this `beta_sft` and exposure level. Any claim that both halves shared
the 120 exposures would be false.

Bank-scenario accuracy can still reveal scenario-specific effects of coded
supervision because the injected prompts and final literal answers correspond
to those scenarios. It provides no convincing evidence of such an effect:

- bank accuracy changed from 81.25% at step 4 to 75.0% at step 150;
- its OLS slope was +0.015 percentage points per step (`r=0.088`), while
  clean accuracy's slope was -0.017 pp/step (`r=-0.142`);
- the bank-minus-clean differential slope was only +0.032 pp/step;
- bank and clean milestone accuracy were essentially unrelated (`r=-0.024`)
  and both fluctuated broadly at their small sample sizes;
- mean bank accuracy was 76.97% over the first 19 milestones and 78.13% over
  the final 20; mean clean accuracy was 78.70% and 78.57%, respectively.

This within-run comparison is descriptive but cannot choose between the two
live explanations: (a) auxiliary CE is a functioning supervised-learning
channel but the model specifically resists adopting non-literal code words, or
(b) the channel is too weak at this `beta_sft`/exposure level to teach either
coded or literal behavior. Neither explanation is favored until the matched
literal-control condition is run. The coded condition's clean negative remains
valid; the mechanism-level diagnosis remains untested.

## Required Step 13b — literal-control condition (not yet run)

**Superseded 2026-08-27: this run is now complete.** The paragraph below is
kept verbatim as the historical record of what was required and why; it is
no longer current. See "2026-08-27 update" near the end of this document for
the completed run, the full 150-step statistical comparison, and the
corrected leading interpretation, which is different from and more specific
than the two explanations this section holds open.

Stage 9 cannot be considered fully closed until the existing 16 literal-control
trajectories are run as the direct minimal-variable comparator: milestone-12
base checkpoint, clean-21 and bank-scenario-16 evaluations, 10% exposure (120
injections over 150 steps), `beta_sft=0.00020350124759158003`, identical
milestone cadence, and unchanged breakers. The only changed variable is
`CODED_TRAJECTORIES` versus the matched literal-control bank. This requirement
is documentation only here; no run was launched.

## Accuracy and KL-clamp trajectory

| step | clean n=21 | bank n=16 | answer-only n=21 | cumulative KL-clamp engagement |
|---:|---:|---:|---:|---:|
| 4 | 80.95% | 81.25% | 42.86% | 14.67% |
| 8 | 85.71% | 81.25% | 28.57% | 21.06% |
| 12 | 76.19% | 68.75% | 23.81% | 20.19% |
| 16 | 80.95% | 81.25% | 42.86% | 22.66% |
| 20 | 80.95% | 87.50% | 33.33% | 22.65% |
| 24 | 71.43% | 75.00% | 33.33% | 24.54% |
| 28 | 76.19% | 87.50% | 38.10% | 23.56% |
| 32 | 71.43% | 75.00% | 47.62% | 24.13% |
| 36 | 80.95% | 75.00% | 28.57% | 25.25% |
| 40 | 80.95% | 62.50% | 38.10% | 25.28% |
| 44 | 80.95% | 81.25% | 38.10% | 25.69% |
| 48 | 76.19% | 68.75% | 28.57% | 25.84% |
| 50 | 80.95% | 68.75% | 28.57% | 25.91% |
| 52 | 76.19% | 68.75% | 38.10% | 25.51% |
| 56 | 85.71% | 75.00% | 38.10% | 25.58% |
| 60 | 76.19% | 87.50% | 33.33% | 26.19% |
| 64 | 71.43% | 81.25% | 38.10% | 25.72% |
| 68 | 76.19% | 81.25% | 33.33% | 25.31% |
| 72 | 85.71% | 75.00% | 38.10% | 26.03% |
| 76 | 80.95% | 87.50% | 38.10% | 26.18% |
| 80 | 85.71% | 75.00% | 47.62% | 26.02% |
| 84 | 80.95% | 68.75% | 33.33% | 26.28% |
| 88 | 85.71% | 81.25% | 38.10% | 26.01% |
| 92 | 85.71% | 68.75% | 33.33% | 25.99% |
| 96 | 76.19% | 68.75% | 33.33% | 26.14% |
| 100 | **90.48%** | 81.25% | 28.57% | 26.66% |
| 104 | 66.67% | 87.50% | 33.33% | 26.33% |
| 108 | 76.19% | 75.00% | 28.57% | 26.20% |
| 112 | 76.19% | 68.75% | 38.10% | 26.62% |
| 116 | 71.43% | 75.00% | 38.10% | 27.01% |
| 120 | 76.19% | **93.75%** | 38.10% | 26.89% |
| 124 | 76.19% | 75.00% | 38.10% | 27.10% |
| 128 | 76.19% | 68.75% | 38.10% | 26.97% |
| 132 | 76.19% | 87.50% | 33.33% | 27.19% |
| 136 | 80.95% | 75.00% | 33.33% | 27.21% |
| 140 | 85.71% | 87.50% | 28.57% | 27.20% |
| 144 | 76.19% | 75.00% | 38.10% | 27.06% |
| 148 | 76.19% | 87.50% | 33.33% | 26.92% |
| 150 | 71.43% | 75.00% | 38.10% | 26.84% |

The run evaluated 39 milestones, not 10: the configured four-step cadence,
the additional required step-50 check, and final step 150.

## Step-100 to step-150 decline

Clean accuracy declined by four examples, from 19/21 (90.48%) to 15/21
(71.43%), a 19.05-point nominal drop. With n=21, one example is worth 4.76
points. Wilson 95% intervals are 71.09-97.35% at step 100 and 50.04-86.19%
at step 150. In the paired scenario comparison, 15 stayed correct, two stayed
wrong, four changed correct-to-wrong, and none changed wrong-to-correct. The
direction is concerning, but exact paired McNemar/binomial `p=0.125` two-sided;
at this evaluation size it is not statistically distinguishable at 95% from
milestone sampling/decoding variation. It is disclosed as a nominal late-run
decline, not asserted as proven task degradation.

## KL-clamp observation

Cumulative clamp engagement rose from 14.67% at step 4, passed 25% at step 36,
peaked at 27.21% at step 136, and ended at 26.84%. Its mean was 24.58%
through step 100 and 26.89% after step 100. This is a small, real and
disclosable upward shift relative to the project's historical 15-25% healthy
range, exceeding its upper edge by about 1.8 points at completion.

It was not a stability event: maximum reported KL was 0.06355 versus the hard
breaker at 5, maximum gradient norm was 30.875 versus 50, and no soft breaker
fired. There is no evidence that the clamp shift caused the accuracy decline.
Across all milestones, clean accuracy and cumulative engagement correlated at
`r=-0.103`; from step 100 onward the correlation was `r=+0.262`, opposite the
proposed negative relationship. Engagement is cumulative, so its early rise
and plateau are not an instantaneous instability measure. The finding is
“modestly elevated engagement without observed destabilization or accuracy
correlation.”

## Checkpoint selection and persistence

Milestone 100 is correctly selected: its 19/21 clean score (90.48%) is the
maximum over the run and selection uses the primary clean set, not the bank
diagnostic. Milestone 120's 15/16 bank peak does not override it. Milestones
100 and 150 are preserved under `aws_runs/bridge_full_10pct_v2/`.

Source-to-local SHA-256 verification:

| artifact | SHA-256 |
|---|---|
| full result JSON | `a8ee30d6198fa6f9d2b01adc38482401f5687364c461932587f233d92273ee74` |
| per-token instrumentation | `c24f4e21c2d01b0014a661a973d4d380848bd2da8bb68f7b3c6fae4b02a5dd89` |
| stdout | `7e270659e1c5e158ff1526bb1b806b31bad86f4fd17be928ca90f0e2cdc9ce6c` |
| milestone-100 adapter | `83d3d1031fef9e849a2a3d40bdfdc183cc95e2054ac55cb6386eae716c6670bb` |
| milestone-150 adapter | `0485d889c466edd3877c51843761cb744d8f73f27efadd874ae26340ac389bd9` |

The source instance was stopped, restored to `g5.xlarge`, and its temporary
SSH ingress rule was removed after these hashes matched.

## 2026-08-27 update — Step 13b complete: full 150-step reanalysis

Step 13b (literal-control) ran to the full 150-step target: `hard_stop=null`,
`soft_stop=null`, `survived_to_full_target=true`. All 39 milestones evaluated
(21 clean + 16 bank each). No NaN/Inf in any of 150 telemetry steps; max
`grad_norm=28.75` (breaker 50); max `kl=0.0608` (breaker 5). Config-verified
identical to Step 13 (coded) except the auxiliary bank content:
`foundation_adapter_sha256`, `loaded_adapter_state_sha256`, `beta_sft
=0.00020350124759158003`, `grad_breaker=50`, `kl_breaker=5`,
`aux_total_examples=120`, milestone cadence, and `adapter_config.json` are all
identical between the two runs' result JSONs.

### Baseline check (requested before drawing any conclusion below)

If the 16 bank scenarios were simply harder than the 21 clean scenarios from
the start, the bank-minus-clean gap should already be negative at step 4,
before meaningful cumulative auxiliary-CE exposure. It is not, for either run:
literal +5.06pp, coded +0.30pp. This rules out a static scenario-difficulty
confound as the sole explanation for the negative gaps reported below.

The first-20-step trajectory is not a clean dose-response ramp, though —
both runs dip negative by step 8-12 and partially recover by step 20:

| step | literal gap | coded gap |
|---:|---:|---:|
| 4 | +5.06pp | +0.30pp |
| 8 | -13.69pp | -4.46pp |
| 12 | -7.44pp | -7.44pp |
| 16 | -1.19pp | +0.30pp |
| 20 | +3.57pp | +6.55pp |

At n=16 (bank) and n=21 (clean) per milestone, one flipped example moves a
single milestone's accuracy by ~6pp, so any one point is noisy; the shape
across the first 20 steps reads as a fast-onset-then-persist pattern rather
than a smooth cumulative buildup.

### Full 39-milestone table, both runs (step 4-150)

| step | lit clean | lit bank | lit gap | cod clean | cod bank | cod gap |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 76.19% | 81.25% | +5.06pp | 80.95% | 81.25% | +0.30pp |
| 8 | 76.19% | 62.50% | -13.69pp | 85.71% | 81.25% | -4.46pp |
| 12 | 76.19% | 68.75% | -7.44pp | 76.19% | 68.75% | -7.44pp |
| 16 | 76.19% | 75.00% | -1.19pp | 80.95% | 81.25% | +0.30pp |
| 20 | 71.43% | 75.00% | +3.57pp | 80.95% | 87.50% | +6.55pp |
| 24 | 76.19% | 68.75% | -7.44pp | 71.43% | 75.00% | +3.57pp |
| 28 | 80.95% | 75.00% | -5.95pp | 76.19% | 87.50% | +11.31pp |
| 32 | 71.43% | 87.50% | +16.07pp | 71.43% | 75.00% | +3.57pp |
| 36 | 76.19% | 75.00% | -1.19pp | 80.95% | 75.00% | -5.95pp |
| 40 | 80.95% | 68.75% | -12.20pp | 80.95% | 62.50% | -18.45pp |
| 44 | 85.71% | 75.00% | -10.71pp | 80.95% | 81.25% | +0.30pp |
| 48 | 80.95% | 75.00% | -5.95pp | 76.19% | 68.75% | -7.44pp |
| 50 | 76.19% | 68.75% | -7.44pp | 80.95% | 68.75% | -12.20pp |
| 52 | 80.95% | 68.75% | -12.20pp | 76.19% | 68.75% | -7.44pp |
| 56 | 66.67% | 62.50% | -4.17pp | 85.71% | 75.00% | -10.71pp |
| 60 | 76.19% | 68.75% | -7.44pp | 76.19% | 87.50% | +11.31pp |
| 64 | 76.19% | 75.00% | -1.19pp | 71.43% | 81.25% | +9.82pp |
| 68 | 71.43% | 81.25% | +9.82pp | 76.19% | 81.25% | +5.06pp |
| 72 | 71.43% | 75.00% | +3.57pp | 85.71% | 75.00% | -10.71pp |
| 76 | 76.19% | 62.50% | -13.69pp | 80.95% | 87.50% | +6.55pp |
| 80 | **85.71%** | 81.25% | -4.46pp | **85.71%** | 75.00% | -10.71pp |
| 84 | 80.95% | 68.75% | -12.20pp | 80.95% | 68.75% | -12.20pp |
| 88 | 71.43% | 68.75% | -2.68pp | 85.71% | 81.25% | -4.46pp |
| 92 | 76.19% | 75.00% | -1.19pp | 85.71% | 68.75% | -16.96pp |
| 96 | 66.67% | 68.75% | +2.08pp | 76.19% | 68.75% | -7.44pp |
| 100 | 71.43% | 62.50% | -8.93pp | **90.48%** | 81.25% | -9.23pp |
| 104 | 71.43% | 68.75% | -2.68pp | 66.67% | 87.50% | **+20.83pp** |
| 108 | 71.43% | 68.75% | -2.68pp | 76.19% | 75.00% | -1.19pp |
| 112 | 80.95% | 81.25% | +0.30pp | 76.19% | 68.75% | -7.44pp |
| 116 | 80.95% | 68.75% | -12.20pp | 71.43% | 75.00% | +3.57pp |
| 120 | 71.43% | 68.75% | -2.68pp | 76.19% | **93.75%** | **+17.56pp** |
| 124 | 80.95% | 68.75% | -12.20pp | 76.19% | 75.00% | -1.19pp |
| 128 | 80.95% | 75.00% | -5.95pp | 76.19% | 68.75% | -7.44pp |
| 132 | 76.19% | 68.75% | -7.44pp | 76.19% | 87.50% | +11.31pp |
| 136 | 76.19% | 68.75% | -7.44pp | 80.95% | 75.00% | -5.95pp |
| 140 | 76.19% | 68.75% | -7.44pp | 85.71% | 87.50% | +1.79pp |
| 144 | 76.19% | 87.50% | +11.31pp | 76.19% | 75.00% | -1.19pp |
| 148 | **85.71%** | 81.25% | -4.46pp | 76.19% | 87.50% | +11.31pp |
| 150 | 66.67% | 68.75% | +2.08pp | 71.43% | 75.00% | +3.57pp |

### Statistics at step 50, step 100, and the full 150

Methodology: gap = bank_acc − clean_acc (percentage points) per milestone.
Sign-flip test = exact enumeration of all sign permutations of the paired
gaps for n≤20, else a 400,000-draw Monte Carlo over sign permutations;
two-sided p = fraction of permuted `|mean|` ≥ observed `|mean|`. Paired
energy-distance test = the univariate energy-distance statistic
`2·E|X−Y| − E|X−X'| − E|Y−Y'|` between the two runs' paired gap sequences,
tested by the matching label-swap permutation scheme (exact for n≤20, else
the same Monte Carlo budget).

| | step 50 (n=13) | step 100 (n=26) | **full 150 (n=39)** |
|---|---:|---:|---:|
| literal sign-flip p (vs. 0) | 0.131 | 0.014 | **0.0013** |
| coded sign-flip p (vs. 0) | 0.330 | 0.059 | **0.478** |
| literal mean gap | -3.73pp | -3.89pp | **-3.91pp** |
| coded mean gap | -2.31pp | -3.35pp | **-1.07pp** |
| paired energy-distance p (coded vs. literal) | 0.573 | 0.686 | **0.102** |
| literal OLS slope | -0.0945 pp/step | -0.0139 pp/step | **-0.0007 pp/step** |
| coded OLS slope | -0.1918 pp/step | -0.0929 pp/step | **+0.0320 pp/step** |

Breaking out the final third (steps 104-150, n=13 each): literal's mean gap
is -3.96pp (consistent with its full-run mean; still mostly negative
milestones), while coded's mean gap is **+3.50pp** (8 of 13 milestones
positive, including three large swings: +20.83pp at step 104, +17.56pp at
step 120, +11.31pp at steps 132 and 148). This is not a single-outlier
artifact.

### Corrected leading interpretation (not a closed conclusion)

Through step 100, coded and literal looked like they shared a similar,
strengthening negative bank-effect, which would have supported a
content-agnostic "auxiliary-injection-causes-degradation" story. The full
150-step data does not bear that out. **Literal's negative bank-minus-clean
gap is real, stable, and persists across the entire run** (sign-flip
`p=0.0013`, near-identical mean gap in both halves). **Coded's negative gap
does not survive the second half** — it reverses to net positive in steps
104-150 and washes out to statistical indistinguishability from zero over
the full run (`p=0.478`). Consistent with this, the two runs move from
indistinguishable from each other through step 100 (energy-distance
`p=0.686`) toward distinguishable over the full run (`p=0.102`).

This is neither of the two explanations Step 13 originally left open
(code-specific resistance vs. an auxiliary channel too weak to teach
anything on either content), and it is also not the simple "both conditions
show the same content-agnostic injection damage" reading that the
through-step-100 numbers alone would have suggested. The step-4 baseline
check rules out a static difficulty confound but does not itself explain why
literal's effect is stable while coded's is not. **This asymmetry is the
current leading open question, disclosed as unresolved by this analysis,**
not a proven mechanism. It should be treated as motivation for a targeted
follow-up (e.g., inspecting per-milestone completions in the reversed window
for coded, steps 104-150) rather than as Stage 9's final word.
