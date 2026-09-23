# Step 8 — milestone-12 token-pair re-audit

Inference-only audit against the actual selected Step-0 milestone-12 LoRA, adapter
SHA-256 `f9f9742cc922cdf45b12f98530862ca941ab0ee0be38b5a22f05cd6de8276080`.
The context and candidate/static eligibility logic were unchanged from Step 4.

| Role | Pair | Base policy | Step-0 milestone 12 | Absolute gap |
|---|---|---|---|---:|
| Training | Nib / Nomo | -6.681682 / -6.681682 | -6.630883 / -6.630883 | 0 |
| Held-out | Yelt / Yark | -7.244182 / -7.244182 | -7.255883 / -7.255883 | 0 |

Both pairs remain statically eligible, both members pass the leakage classifier, and
all four tokens retain minimum edit distance 3 from banned vocabulary. The Step-0
policy moved both members of each pair by exactly the same amount, so the original
within-pair frequency control is preserved and selection does not need revisiting on
this ground.

Step 9 was not executed. Machine-readable evidence is in `results.json`; captured
execution output is in `stdout.log`.
