# Phase 5 v2: noise-aware analysis (3 seeds)

All intervals are 95% Student-t (2 d.f.). Positive gap = degradation.

## Per-probe verdicts

| method | metric | mean | sd | 95% CI | verdict |
|---|---|---|---|---|---|
| lora | samsum_rl_gain | 0.1019 | 0.0069 | [0.0849, 0.1190] |  |
| lora | boolq_forgetting | -0.0260 | 0.0106 | [-0.0523, 0.0003] | no detectable change |
| lora | piqa_forgetting | -0.0147 | 0.0042 | [-0.0250, -0.0043] | improvement |
| lora | arc_forgetting | 0.0067 | 0.0070 | [-0.0108, 0.0241] | no detectable change |
| lora | mmlu_forgetting | -0.0033 | 0.0140 | [-0.0382, 0.0316] | no detectable change |
| lora | wikitext_ppl_increase | 0.9453 | 0.0672 | [0.7783, 1.1123] | degradation |
| adalora | samsum_rl_gain | 0.0504 | 0.0014 | [0.0469, 0.0539] |  |
| adalora | boolq_forgetting | -0.0127 | 0.0081 | [-0.0327, 0.0074] | no detectable change |
| adalora | piqa_forgetting | 0.0007 | 0.0050 | [-0.0118, 0.0132] | no detectable change |
| adalora | arc_forgetting | -0.0047 | 0.0046 | [-0.0161, 0.0068] | no detectable change |
| adalora | mmlu_forgetting | 0.0060 | 0.0035 | [-0.0026, 0.0146] | no detectable change |
| adalora | wikitext_ppl_increase | 0.0010 | 0.0089 | [-0.0211, 0.0230] | no detectable change |
| dora | samsum_rl_gain | 0.1003 | 0.0086 | [0.0788, 0.1218] |  |
| dora | boolq_forgetting | -0.0260 | 0.0087 | [-0.0477, -0.0043] | improvement |
| dora | piqa_forgetting | -0.0127 | 0.0042 | [-0.0230, -0.0023] | improvement |
| dora | arc_forgetting | 0.0140 | 0.0092 | [-0.0088, 0.0368] | no detectable change |
| dora | mmlu_forgetting | -0.0013 | 0.0172 | [-0.0442, 0.0415] | no detectable change |
| dora | wikitext_ppl_increase | 0.9957 | 0.0632 | [0.8387, 1.1527] | degradation |

## Significant paired differences (unadjusted p<.05; Holm-adjusted p shown)

| a - b | metric | mean diff | 95% CI | p | p_holm |
|---|---|---|---|---|---|
| lora - adalora | samsum_rl_gain | 0.0515 | [0.0318, 0.0712] | 0.0078 | 0.1090 |
| lora - adalora | piqa_forgetting | -0.0153 | [-0.0229, -0.0077] | 0.0130 | 0.1578 |
| lora - adalora | wikitext_ppl_increase | 0.9443 | [0.7961, 1.0926] | 0.0013 | 0.0226 |
| lora - dora | arc_forgetting | -0.0073 | [-0.0131, -0.0016] | 0.0315 | 0.3465 |
| lora - dora | wikitext_ppl_increase | -0.0504 | [-0.0680, -0.0329] | 0.0065 | 0.0972 |
| adalora - dora | samsum_rl_gain | -0.0498 | [-0.0737, -0.0260] | 0.0121 | 0.1578 |
| adalora - dora | piqa_forgetting | 0.0133 | [0.0105, 0.0162] | 0.0025 | 0.0399 |
| adalora - dora | wikitext_ppl_increase | -0.9948 | [-1.1343, -0.8552] | 0.0011 | 0.0191 |

## Significance-aware Pareto (adaptation vs WikiText PPL increase)

| method | efficient | dominated by |
|---|---|---|
| lora | True | - |
| adalora | True | - |
| dora | False | lora |

## Composite winners by retention weight (fixed-reference scaling)

| ref scale | method | weight range where it wins | #grid points |
|---|---|---|---|
| x0.5 | adalora | 0.72-1.00 | 29 |
| x0.5 | lora | 0.00-0.71 | 72 |
| x1.0 | adalora | 0.83-1.00 | 18 |
| x1.0 | lora | 0.00-0.82 | 83 |
| x2.0 | adalora | 0.91-1.00 | 10 |
| x2.0 | lora | 0.00-0.90 | 91 |
