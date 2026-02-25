# Local Patches for DiffusionRL Integration

## Scope (as of 2026-02-25)

- Upstream base: `43f83525c` (`origin/main`)
- Local branch: `local/diffusion-rl`
- Delta from upstream: 2 commits (`origin/main..local/diffusion-rl`)
- Runtime patch commit: `2c3971f77`
- Patch index/docs commit: `c584076ef`

`2c3971f77` runtime patch intent:
- Merge PR #18806: rollout SDE/CPS per-step `log_prob`
- Merge PR #19153: `release_memory_occupation()` / `resume_memory_occupation()`

## Applied PRs

| PR | Author | Summary | Local status |
|----|--------|---------|--------------|
| [#18806](https://github.com/sgl-project/sglang/pull/18806) | MikukuOvO | Flow-matching SDE/CPS per-step `log_prob` computation | Cherry-picked into `2c3971f77` |
| [#19153](https://github.com/sgl-project/sglang/pull/19153) | Godmook | GPU sleep/wake via `release_memory_occupation()` / `resume_memory_occupation()` | Cherry-picked into `2c3971f77` |

## Upstream Tracking Snapshot (2026-02-25)

- Upstream main after fetch: `origin/main = d0bb14003`
- Branch divergence (local patch branch vs upstream main):
  - `origin/main..local/diffusion-rl`: 2 commits
  - `local/diffusion-rl..origin/main`: 92 commits
- GitHub PR states at check time:
  - `#18806`: **open**, not merged, head `dfb3b0c5868bcdef6f0eb8752a7628577a64b7ac`, updated `2026-02-24T21:56:35Z`
  - `#19153`: **open**, not merged, head `99d943ed6a0dff3de7b57057cdd346cdc2c6afe5`, updated `2026-02-23T20:09:21Z`
- Code-level check against current `origin/main`: key symbols still absent on upstream main
  - `rollout_sde_type`
  - `trajectory_log_probs`
  - `/release_memory_occupation`
  - `/resume_memory_occupation`

Notes:
- No auto-rebase/drop performed because neither tracked PR is merged yet.
- If either PR merges, re-run the verification block below and then rebase `local/diffusion-rl` onto latest `origin/main`.

## Runtime Patch Manifest (git-numstat from `2c3971f77`)

| File | + | - | What changed |
|------|---|---|--------------|
| `python/sglang/multimodal_gen/configs/sample/sampling_params.py` | 25 | 0 | Added rollout params: `rollout`, `rollout_sde_type`, `rollout_noise_level` and CLI args |
| `python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py` | 37 | 0 | Added `release_memory_occupation()` / `resume_memory_occupation()` client APIs; added `trajectory_log_probs` pass-through |
| `python/sglang/multimodal_gen/runtime/entrypoints/http_server.py` | 74 | 0 | Added `/release_memory_occupation` and `/resume_memory_occupation` endpoints |
| `python/sglang/multimodal_gen/runtime/entrypoints/openai/image_api.py` | 3 | 0 | Forward rollout params from OpenAI image request |
| `python/sglang/multimodal_gen/runtime/entrypoints/openai/protocol.py` | 6 | 0 | Added rollout fields to image/video request models |
| `python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py` | 9 | 0 | Forward rollout params from OpenAI video request/form data |
| `python/sglang/multimodal_gen/runtime/entrypoints/utils.py` | 15 | 0 | Added `ReleaseMemoryOccupationReq`, `ResumeMemoryOccupationReq`; added `GenerationResult.trajectory_log_probs` |
| `python/sglang/multimodal_gen/runtime/managers/gpu_worker.py` | 148 | 0 | Implemented weight offload/onload behavior for sleep/wake (incl. layerwise offload path) |
| `python/sglang/multimodal_gen/runtime/managers/scheduler.py` | 61 | 0 | Added request handlers for sleep/wake; added `is_sleeping` guard to reject generation while sleeping |
| `python/sglang/multimodal_gen/runtime/pipelines/patches/__init__.py` | 1 | 0 | New patch package marker |
| `python/sglang/multimodal_gen/runtime/pipelines/patches/flow_matching_with_logprob.py` | 115 | 0 | New SDE/CPS rollout step with per-step logprob output |
| `python/sglang/multimodal_gen/runtime/pipelines_core/schedule_batch.py` | 2 | 0 | Added `trajectory_log_probs` field in `Req` and `OutputBatch` |
| `python/sglang/multimodal_gen/runtime/pipelines_core/stages/decoding.py` | 1 | 0 | Pass-through `trajectory_log_probs` |
| `python/sglang/multimodal_gen/runtime/pipelines_core/stages/decoding_av.py` | 1 | 0 | Pass-through `trajectory_log_probs` |
| `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py` | 58 | 7 | Integrated rollout branch in denoising loop; hook scheduler with `sde_step_with_logprob`; collect/publish per-step logprobs |
| `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_dmd.py` | 1 | 0 | Threaded empty `trajectory_log_probs` through DMD path |

Runtime patch totals (`2c3971f77`): **+557 / -7**, 16 files changed.

## Added/Changed Interfaces

### Request/response surface

- Sampling fields:
  - `rollout: bool`
  - `rollout_sde_type: "sde" | "cps"`
  - `rollout_noise_level: float`
- Output field:
  - `trajectory_log_probs`

### Memory control APIs

- Python:
  - `DiffGenerator.release_memory_occupation(tags: List[str] | None)`
  - `DiffGenerator.resume_memory_occupation(tags: List[str] | None)`
- HTTP:
  - `POST /release_memory_occupation`
  - `POST /resume_memory_occupation`
- Scheduler behavior:
  - Maintains `is_sleeping`
  - Rejects generation requests while sleeping

## Compatibility Notes for DiffusionRL

- This branch provides the exact capabilities DiffusionRL expects for:
  - checkpoint-path weight update (`UpdateWeightFromDiskReqInput`)
  - optional native rollout logprob (`trajectory_log_probs`)
  - rollout/training memory swap (`release/resume_memory_occupation`)
- Trajectory currently remains denoising outputs only (`[x_{T-1}, ..., x_0]`) in `denoising.py`; initial `x_T` is not prepended by this patch.

## Known Issues / Limitations

1. `t.item()` in `flow_matching_with_logprob.py` may introduce per-step CPU-GPU synchronization overhead.
2. `release/resume_memory_occupation(tags=...)` currently only handles `"weights"` for diffusion.
3. No dedicated tests were added in this local patch commit; validation relies on integration scripts/runtime checks.

## Verification Commands

```bash
# Confirm branch and runtime patch commits
git branch --show-current
git log --oneline origin/main..local/diffusion-rl

# Confirm runtime patch file set
git show --name-status --format='' 2c3971f77
git show --numstat --format='' 2c3971f77
```

## Rollback Condition

When both upstream PRs are merged into `origin/main` and validated locally:

```bash
git checkout main
git pull origin main
git branch -d local/diffusion-rl
```
