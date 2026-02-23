# Local Patches for DiffusionRL Integration

## Base Commit

```
43f83525c Revert "[AMD] support two batch overlapping for mori ep #17953" (#19161)
```

Branch: `main` (upstream: `sgl-project/sglang`)

## Applied PRs

| PR | Author | Summary | Status |
|----|--------|---------|--------|
| [#18806](https://github.com/sgl-project/sglang/pull/18806) | MikukuOvO | Flow-matching SDE/CPS per-step `log_prob` computation | Cherry-picked |
| [#19153](https://github.com/sgl-project/sglang/pull/19153) | Godmook | `release_memory_occupation()` / `resume_memory_occupation()` for GPU sleep/wake | Cherry-picked |

## Files Changed

- `python/sglang/multimodal_gen/configs/sample/sampling_params.py` — rollout params (`rollout`, `rollout_sde_type`, `rollout_noise_level`)
- `python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py` — `release/resume_memory_occupation` API
- `python/sglang/multimodal_gen/runtime/entrypoints/http_server.py` — HTTP endpoints for sleep/wake + weight update
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/` — OpenAI-compatible API extensions
- `python/sglang/multimodal_gen/runtime/entrypoints/utils.py` — `ReleaseMemoryOccupationReq` / `ResumeMemoryOccupationReq`
- `python/sglang/multimodal_gen/runtime/managers/gpu_worker.py` — weight offload/onload + `update_weights_from_disk`
- `python/sglang/multimodal_gen/runtime/managers/scheduler.py` — scheduler dispatch for new request types
- `python/sglang/multimodal_gen/runtime/pipelines/patches/` — **NEW** SDE/CPS log_prob math
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py` — rollout log_prob integration in denoising loop

## Known Issues

1. **GPU sync overhead**: `t.item()` in the log_prob path causes CPU-GPU synchronization per timestep. Acceptable for RL training but not ideal for high-throughput serving.
2. **tags limitation**: `release_memory_occupation(tags=...)` only supports `"weights"`. KV-cache and activations are not yet managed.

## Rollback Condition

When **both** PRs are merged to upstream `main`:
```bash
git checkout main
git pull origin main
git branch -d local/diffusion-rl
```
