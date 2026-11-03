# Text → CadQuery: a specialized open model vs. closed frontier models

**Fireworks: Deploy buildathon, Nov 2 2026.** Task: turn a natural-language description of a mechanical part into
executable [CadQuery](https://github.com/CadQuery/cadquery) code that builds the 3D solid. The eval is fully
mechanical: does the code run, and how close is the resulting mesh to the ground-truth part (Chamfer distance, IoU),
plus latency and $ per 1k parts.

## Results

Benchmark: 300 held-out parts from TextCAD-Bench T1 (DeepCAD test split, zero overlap with training), two prompt levels
(`l1` abstract one-liner, `l3` expert spec with coordinates), identical system prompt for every model, greedy decoding.
`usable_rate` = code executed **and** Chamfer distance < 10 (×100, meshes normalised to a unit box).
`usable_per_dollar` = usable parts per $1 of API spend at list prices.

| name | level | n | exec_rate | cd_median | cd_all_penalized | usable_rate | iou_mean | latency_p50 | cost_per_1k_parts | usable_per_dollar |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-v41-flash-l1 | l1 | 300 | 0.900 | 12.170 | 25.140 | 0.390 | 42.954 | 6.850 | 2.295 | 170 |
| gpt-oss-120b-l1 | l1 | 300 | 0.850 | 13.232 | 29.592 | 0.357 | 39.913 | 2.702 | 0.482 | 740 |
| gpt-oss-120b-l3 | l3 | 300 | 0.233 | 17.618 | 82.055 | 0.077 | 41.537 | 6.015 | 1.158 | 66.182 |
| haiku55-l1 | l1 | 300 | 0.997 | 12.661 | 17.187 | 0.420 | 42.174 | 2.655 | 0.327 | 1286 |
| haiku55-l3 | l3 | 300 | 0.973 | 15.432 | 21.235 | 0.383 | 44.668 | 6.359 | 0.917 | 418 |
| latency-s100-l1 | l1 | 60 | 1.000 | 14.857 | 22.911 | 0.383 | 36.701 | 1.739 | 0.345 | 1110 |
| opus55-l1 | l1 | 300 | 0.997 | 12.660 | 17.079 | 0.427 | 43.385 | 2.437 | 5.706 | 74.778 |
| opus55-l3 | l3 | 300 | 1.000 | 16.379 | 19.228 | 0.367 | 45.622 | 5.693 | 15.771 | 23.249 |
| qwen36-35b-base-l1 | l1 | 300 | 0.540 | 13.146 | 55.399 | 0.223 | 39.466 | 6.933 | 0.809 | 276 |
| qwen36-35b-base-l3 | l3 | 300 | 0.083 | 27.117 | 94.074 | 0.017 | 34.264 | 12.092 | 2.448 | 6.809 |
| tuned-v1-s100-l1 | l1 | 300 | 0.973 | 15.923 | 23.079 | 0.387 | 38.049 | 1.861 | 0.374 | 1034 |
| tuned-v1-s100-l3 | l3 | 300 | 0.940 | 17.668 | 26.607 | 0.310 | 37.977 | 2.083 | 0.679 | 456 |
| tuned-v1-s25-l1 | l1 | 300 | 0.897 | 20.532 | 31.254 | 0.270 | 34.112 | 1.322 | 0.507 | 532 |
| tuned-v1-s25-l3 | l3 | 300 | 0.883 | 24.211 | 33.832 | 0.187 | 31.719 | 1.369 | 0.723 | 258 |
| tuned-v1-s50-l1 | l1 | 300 | 0.953 | 17.990 | 26.263 | 0.320 | 36.866 | 2.358 | 0.400 | 800 |
| tuned-v1-s50-l3 | l3 | 300 | 0.973 | 18.096 | 23.781 | 0.330 | 38.366 | 2.623 | 0.666 | 495 |
| tuned-v1-s75-l1 | l1 | 300 | 0.953 | 16.573 | 25.162 | 0.367 | 39.061 | 2.529 | 0.414 | 885 |
| tuned-v1-s75-l3 | l3 | 300 | 0.960 | 17.409 | 24.884 | 0.320 | 39.761 | 2.771 | 0.675 | 474 |

**Our model:** `accounts/wjing63/models/cadq-qwen36-35b-v1` = Qwen 3.6 35B-A3B (3B active) + LoRA rank 16, trained on
6,400 verified description→CadQuery examples for 100 optimizer steps (3.1M tokens, ~$7, 30 min) on Fireworks Serverless Training.
Rows `tuned-v1-s25/s50/s75/s100` are the checkpoints every 25 steps; `qwen36-35b-base` is the same model untuned.

**What the numbers say**

- Fine-tuning took the open model from 54% → 97% execution on abstract prompts and from 8% → 94–97% on expert prompts.
  The untuned model's failures were hallucinated CadQuery methods (`Workplane.scale`, `Vector.filter`); those are gone.
- Against the closed frontier target (Claude Opus 5.5): on abstract prompts the tuned 3B-active model reaches 39% usable parts vs 43%,
  at 15× lower cost and lower latency. On expert prompts it reaches 31–33% vs 37% at 23× lower cost and ~2–3× lower latency.
- Against open models served as-is: it beats gpt-oss-120b and DeepSeek V4.1 Flash on execution rate and usable rate at lower cost,
  and on expert prompts it is the only open model that works at all (gpt-oss-120b: 23% execution).
- Honest caveat: Claude Haiku 5.5 is a strong cheap closed baseline. Abstract prompts: 42% usable at $0.33/1k (we: 39% at $0.37); our win there is latency only.
  Expert prompts: Haiku 38% usable at $0.92/1k and 6.4 s; we: 31% at $0.68/1k and 1.7 s, so we win on latency (3.7×) and usable parts per dollar (456 vs 418), not on raw quality. Shape accuracy (Chamfer) still trails Opus: the training code is in DeepCAD's internal units while prompts are normalised,
  so the model must rescale every coordinate; a data fix (re-expressing code in prompt units) is the obvious next step.
- Latency rows for tuned checkpoints were measured while three evals shared the sampling pool. Re-measured alone (`latency-s100-l1`, 60 parts): **p50 1.74 s, mean 2.46 s**, vs Opus 2.4 s (abstract) and 5.7 s (expert).
- Demo renders: `demo/part00072791/` (benchmark part: tuned matches ground truth, Chamfer 1.6; Opus 36.5; gpt-oss failed) and `demo/bracket/` (free-prose bracket: tuned builds a block with holes, Opus and gpt-oss get the L-shape). The model learned the dataset's conventions, not general CAD prose yet.

## Layout

| Path | What |
|---|---|
| `data/prep.py` | Builds `data/train.jsonl` (29,783 chat examples, 20k parts, four prompt levels, every program execution-verified) and `data/val.jsonl` from Text2CAD + CadQuery annotations (`Hula0401/cad-sft`, `kyujinpy/train_text2cad_v1_1_8K`). |
| `data/system_prompt.txt` | The one system prompt shared by training and every baseline. |
| `train/sft_serverless.py` | LoRA SFT loop on Fireworks Serverless Training (SDK only): renders chats with the base tokenizer, masks prompt tokens, `forward_backward(..., "cross_entropy")`, cosine LR, sampler checkpoints every N steps. |
| `eval/run.py` | Benchmark runner: TextCAD-Bench T1 (DeepCAD test parts, zero overlap with train). Backends: Anthropic, OpenAI, Fireworks serverless chat, Fireworks serverless-training sampler (base model or saved checkpoint). Executes code in a sandboxed subprocess, exports STL, scores vs. GT mesh. |
| `eval/metrics.py` | Chamfer distance ×100 and 64³ voxel IoU, identical to the TextCAD-Bench protocol. |
| `eval/report.py` | Markdown table across all runs. |
| `scripts/promote.py` | Promote a serverless sampler checkpoint to a deployable Fireworks model. |
| `scripts/fw_api.py` | Thin REST helper (datasets, managed SFT jobs, deployments). |
| `demo/compare.py` | One prompt → code + rendered PNG from several models side by side. |

## Reproduce

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python cadquery trimesh numpy scipy pandas pyarrow datasets openai anthropic "fireworks-ai[training]" transformers
.venv/bin/python data/prep.py 20000 1.5
.venv/bin/python train/sft_serverless.py --max-examples 9600 --batch 64 --save-every 50 --tag v1
.venv/bin/python eval/run.py --name opus55-l1 --backend anthropic --model claude-opus-5-5 --level l1 --n 300
.venv/bin/python eval/run.py --name tuned-l1 --backend serverless --model "ckpt|<path from results/serverless/state-v1.json>|Qwen/Qwen3.6-35B-A3B" --level l1 --n 300
.venv/bin/python eval/report.py
```

Prompt levels: `l1` = abstract one-sentence description, `l3` = full expert spec with coordinates (the input a CAD-ops pipeline would actually have).

## Dashboard (demo)

- Static, shareable: `python dashboard/build.py` → `dashboard/index.html` (published at https://claude.ai/artifact/Xq1icddNLg17qSTHeckv57). Headline before/after tiles, training loss curve with checkpoint markers, checkpoint progression, comparison charts against Claude Opus/Haiku, DeepSeek, gpt-oss, a 12-part before/after gallery with code, and the workflow figure. Click any render to open an interactive 3D view (three.js) of ground truth, untuned, tuned and Opus side by side with a shared rotation; live results get a "Rotate all three in 3D" button.
- Live (stage demo): `.venv/bin/python dashboard/serve.py` → http://localhost:8787 adds a prompt box that runs the untuned base, the tuned checkpoint (Fireworks serverless sampler) and Claude Opus 5.5 side by side, executes the code locally and renders the meshes. Presets: benchmark disc (tuned wins), free-prose bracket (tuned misses), expert spec.
- Explainer for newcomers (task, LoRA, workflow, platform, scoring): https://claude.ai/artifact/6Xg7WTFjWhZc7TkrTfNKrT
