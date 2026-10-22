# Phase 5 v2: noise-aware analysis (3 seeds)

All intervals are 95% Student-t (2 d.f.). Positive gap = degradation.

## Per-probe verdicts

| method | metric | mean | sd | 95% CI | verdict |
|---|---|---|---|---|---|
| lora | samsum_rl_gain | 0.0987 | 0.0012 | [0.0959, 0.1016] |  |
| lora | boolq_forgetting | -0.0213 | 0.0130 | [-0.0537, 0.0110] | no detectable change |
| lora | piqa_forgetting | -0.0167 | 0.0031 | [-0.0243, -0.0091] | improvement |
| lora | arc_forgetting | 0.0107 | 0.0117 | [-0.0184, 0.0398] | no detectable change |
| lora | mmlu_forgetting | -0.0007 | 0.0061 | [-0.0158, 0.0145] | no detectable change |
| lora | wikitext_ppl_increase | 0.8797 | 0.0556 | [0.7416, 1.0177] | degradation |
| adalora | samsum_rl_gain | 0.0542 | 0.0117 | [0.0252, 0.0832] |  |
| adalora | boolq_forgetting | -0.0027 | 0.0081 | [-0.0227, 0.0174] | no detectable change |
| adalora | piqa_forgetting | 0.0000 | 0.0040 | [-0.0099, 0.0099] | no detectable change |
| adalora | arc_forgetting | 0.0000 | 0.0053 | [-0.0131, 0.0131] | no detectable change |
| adalora | mmlu_forgetting | 0.0007 | 0.0050 | [-0.0118, 0.0132] | no detectable change |
| adalora | wikitext_ppl_increase | -0.0039 | 0.0016 | [-0.0079, 0.0001] | no detectable change |
| dora | samsum_rl_gain | 0.1036 | 0.0068 | [0.0866, 0.1206] |  |
| dora | boolq_forgetting | -0.0193 | 0.0101 | [-0.0443, 0.0057] | no detectable change |
| dora | piqa_forgetting | -0.0147 | 0.0061 | [-0.0298, 0.0005] | no detectable change |
| dora | arc_forgetting | 0.0113 | 0.0129 | [-0.0206, 0.0433] | no detectable change |
| dora | mmlu_forgetting | 0.0000 | 0.0111 | [-0.0277, 0.0277] | no detectable change |
| dora | wikitext_ppl_increase | 0.9717 | 0.0422 | [0.8668, 1.0766] | degradation |

## Significant paired differences (unadjusted p<.05; Holm-adjusted p shown)

| a - b | metric | mean diff | 95% CI | p | p_holm |
|---|---|---|---|---|---|
| lora - adalora | samsum_rl_gain | 0.0445 | [0.0151, 0.0740] | 0.0229 | 0.2971 |
| lora - adalora | piqa_forgetting | -0.0167 | [-0.0195, -0.0138] | 0.0016 | 0.0255 |
| lora - adalora | wikitext_ppl_increase | 0.8836 | [0.7489, 1.0182] | 0.0013 | 0.0213 |
| adalora - dora | samsum_rl_gain | -0.0494 | [-0.0662, -0.0326] | 0.0062 | 0.0933 |
| adalora - dora | piqa_forgetting | 0.0147 | [0.0089, 0.0204] | 0.0082 | 0.1143 |
| adalora - dora | wikitext_ppl_increase | -0.9756 | [-1.0767, -0.8746] | 0.0006 | 0.0104 |

## Significance-aware Pareto (adaptation vs WikiText PPL increase)

| method | efficient | dominated by |
|---|---|---|
| lora | True | - |
| adalora | True | - |
| dora | True | - |

## Composite winners by retention weight (fixed-reference scaling)

| ref scale | method | weight range where it wins | #grid points |
|---|---|---|---|
| x0.5 | adalora | 0.71-1.00 | 30 |
| x0.5 | dora | 0.00-0.70 | 71 |
| x1.0 | adalora | 0.81-1.00 | 20 |
| x1.0 | dora | 0.00-0.80 | 81 |
| x2.0 | adalora | 0.90-1.00 | 11 |
| x2.0 | dora | 0.00-0.89 | 90 |
