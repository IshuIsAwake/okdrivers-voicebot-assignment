# Ablation Summary (median across test clips)

| stt_engine   | llm_engine   | tts_engine   |   e2e_latency_sec |   stt_rtf |   llm_ttft_sec |   llm_tps |   tts_rtf |   wer |   stt_peak_mem_mb |   tts_peak_mem_mb |
|:-------------|:-------------|:-------------|------------------:|----------:|---------------:|----------:|----------:|------:|------------------:|------------------:|
| fw_base      | gemma4_cloud | edge_tts     |             2.459 |     0.023 |          0.718 |    25.714 |     0.387 | 1.000 |           426.418 |           426.465 |
| fw_base      | gemma4_cloud | piper        |             1.283 |     0.034 |          0.758 |    20.868 |     0.023 | 1.000 |           431.152 |           525.000 |
| fw_base      | gemma4_local | edge_tts     |             2.281 |     0.033 |          0.199 |    23.065 |     0.374 | 1.000 |           385.789 |           386.035 |
| fw_base      | gemma4_local | piper        |             0.989 |     0.036 |          0.225 |    19.862 |     0.029 | 1.000 |           379.695 |           430.676 |
| fw_base      | groq_8b      | edge_tts     |             1.986 |     0.023 |          0.178 |   110.742 |     0.341 | 1.000 |           364.156 |           364.828 |
| fw_base      | groq_8b      | piper        |             0.583 |     0.033 |          0.292 |    69.871 |     0.026 | 1.000 |           325.547 |           459.023 |
| fw_small     | gemma4_cloud | edge_tts     |             2.014 |     0.086 |          0.728 |    19.363 |     0.460 | 1.571 |           841.414 |           841.734 |
| fw_small     | gemma4_cloud | piper        |             1.577 |     0.092 |          0.756 |    16.097 |     0.024 | 1.571 |           715.199 |           816.637 |
| fw_small     | gemma4_local | edge_tts     |             2.350 |     0.086 |          0.194 |    22.366 |     0.353 | 1.571 |           812.852 |           812.902 |
| fw_small     | gemma4_local | piper        |             1.129 |     0.092 |          0.201 |    20.435 |     0.035 | 1.571 |           724.961 |           755.742 |
| fw_small     | groq_8b      | edge_tts     |             2.484 |     0.086 |          0.346 |    88.376 |     0.284 | 1.571 |           803.512 |           805.406 |
| fw_small     | groq_8b      | piper        |             0.588 |     0.092 |          0.149 |   131.922 |     0.024 | 1.571 |           697.191 |           800.352 |

## Best per metric
- **e2e_latency_sec** (lower better): fw_base / groq_8b / piper = 0.583
- **wer** (lower better): fw_base / gemma4_cloud / edge_tts = 1.000
- **stt_peak_mem_mb** (lower better): fw_base / groq_8b / piper = 325.547
- **tts_peak_mem_mb** (lower better): fw_base / groq_8b / edge_tts = 364.828
- **llm_tps** (higher better): fw_small / groq_8b / piper = 131.922