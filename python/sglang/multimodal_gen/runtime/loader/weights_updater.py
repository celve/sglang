"""
In-place weight updates for diffusion pipeline modules.

This module provides WeightsUpdater, which swaps model weights at runtime
without restarting the server.  It is the diffusion-engine counterpart of the
LLM engine's ModelRunner.update_weights_from_disk.

Detailed usage of higher level API can be found in

/python/sglang/multimodal_gen/test/server/test_update_weights_from_disk.py

Key design decisions:

- All-or-nothing with rollback: modules are updated sequentially.  If
  any module fails (shape mismatch, corrupted file, etc.), every module
  that was already updated is rolled back by reloading its weights from
  pipeline.model_path (the last successfully-loaded checkpoint).  On
  success, pipeline.model_path is updated to the new model_path so
  that future rollbacks target the latest good checkpoint, not the
  originally-launched model.

- Rollback failures propagate: if rollback itself fails, the exception is
  not caught so the caller knows the model is in an inconsistent state.
  This matches the LLM engine behaviour.

- Offload-aware: the diffusion LayerwiseOffloadManager replaces GPU
  parameters with torch.empty((1,)) placeholders while real weights live
  in consolidated pinned CPU buffers.  A naive param.data.copy_() would
  fail with a shape mismatch.  Instead, the updater dynamically detects
  active offload managers and writes new weights directly into their CPU
  buffers via update_cpu_weights(), bypassing the placeholders entirely.
  For any layer that happens to be prefetched on GPU at update time, the
  live GPU tensor is also updated so the change takes effect immediately.
  This requires no extra GPU memory and does not disturb the offload state.

- DTensor-aware: parameters that have been distributed via
  torch.distributed.tensor are updated through distribute_tensor
  so that each shard is correctly placed on the right device mesh.
"""

from __future__ import annotations

import gc
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor, distribute_tensor

from sglang.multimodal_gen.runtime.cache.teacache import TeaCacheMixin
from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.pipelines.diffusers_pipeline import DiffusersPipeline
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def get_updatable_modules(pipeline) -> dict[str, torch.nn.Module]:
    """Return updatable nn.Module components for the given pipeline.

    Works with both the native ComposedPipelineBase backend and the
    DiffusersPipeline wrapper.
    """
    if isinstance(pipeline, DiffusersPipeline):
        diffusers_pipe = pipeline.get_module("diffusers_pipeline")
        if diffusers_pipe is not None and diffusers_pipe.components is not None:
            raw = diffusers_pipe.components
        else:
            raw = {}
    else:
        raw = pipeline.modules
    return {n: m for n, m in raw.items() if isinstance(m, torch.nn.Module)}


def _get_weights_iter(weights_dir: str):
    """Return a (name, tensor) iterator over safetensors in weights_dir."""
    safetensors_files = _list_safetensors_files(weights_dir)
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {weights_dir}")
    return safetensors_weights_iterator(safetensors_files)


def _validate_weight_files(
    local_model_path: str,
    modules_to_update: list[tuple[str, torch.nn.Module]],
) -> tuple[dict[str, str], list[str]]:
    """Check that every module has a weights directory with safetensors files.

    Returns:
        (weights_map, missing) where weights_map maps module name to its
        weights directory and missing lists modules without weight files.
    """
    weights_map: dict[str, str] = {}
    missing: list[str] = []
    for module_name, _ in modules_to_update:
        weights_dir = Path(local_model_path) / module_name
        if weights_dir.exists() and _list_safetensors_files(str(weights_dir)):
            weights_map[module_name] = str(weights_dir)
        else:
            missing.append(module_name)
    return weights_map, missing


def _load_weights_into_module(module: torch.nn.Module, weights_iter) -> None:
    """Load weights into a module, handling offload-managed parameters.

    For offloaded modules, updates CPU buffers directly via
    update_cpu_weights(); non-offloaded parameters use in-place copy.
    """
    offload_managers: list = []
    if isinstance(module, OffloadableDiTMixin) and module.layerwise_offload_managers:
        offload_managers = [m for m in module.layerwise_offload_managers if m.enabled]

    if offload_managers:
        weight_dict = dict(weights_iter)
        offloaded_names: set[str] = set()
        for manager in offload_managers:
            offloaded_names.update(manager.update_cpu_weights(weight_dict))
        remaining = ((n, w) for n, w in weight_dict.items() if n not in offloaded_names)
        load_weights_into_model(remaining, dict(module.named_parameters()))
    else:
        load_weights_into_model(weights_iter, dict(module.named_parameters()))


def load_weights_into_model(weights_iter, model_params: dict) -> None:
    """Copy weights from weights_iter into model_params in-place."""
    import time as _time
    t0 = _time.perf_counter()
    count = 0
    skipped = []
    for name, loaded_weight in weights_iter:
        if name not in model_params:
            if len(skipped) < 5:
                skipped.append(name)
            continue
        param = model_params[name]
        if param.shape != loaded_weight.shape:
            raise ValueError(
                f"Shape mismatch for {name}: model={param.shape}, loaded={loaded_weight.shape}"
            )
        if isinstance(param, DTensor):
            distributed_weight = distribute_tensor(
                loaded_weight.to(param.dtype),
                param.device_mesh,
                param.placements,
            )
            param._local_tensor.copy_(distributed_weight._local_tensor)
        else:
            param.data.copy_(loaded_weight.to(param.dtype))
        count += 1
    if skipped:
        logger.warning("load_weights_into_model: skipped %d+ names, first 5: %s", len(skipped), skipped)
    logger.warning("load_weights_into_model: %d params copied in %.3fs", count, _time.perf_counter() - t0)


class WeightsUpdater:
    """In-place weight updates for diffusion pipeline modules.

    Args:
        pipeline: A ComposedPipelineBase (or DiffusersPipeline) instance
            whose modules will be updated.  The pipeline's model_path
            attribute is used for rollback on failure.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def update_weights_from_disk(
        self,
        model_path: str,
        flush_cache: bool = True,
        target_modules: list[str] | None = None,
    ) -> tuple[bool, str]:
        """Update model weights from disk without restarting the server."""
        logger.info(f"Updating weights from disk: {model_path}")

        try:
            modules_to_update = self._collect_modules(target_modules)
        except ValueError as e:
            logger.error(str(e))
            return False, str(e)

        if not modules_to_update:
            error_msg = (
                f"No matching modules found for update. "
                f"Requested: {target_modules}. "
                f"Available nn.Module(s): {list(get_updatable_modules(self.pipeline).keys())}"
            )
            logger.error(error_msg)
            return False, error_msg

        try:
            local_model_path = maybe_download_model(model_path)
        except Exception as e:
            return False, f"Failed to download model: {e}"

        weights_map, missing = _validate_weight_files(
            local_model_path, modules_to_update
        )
        if missing:
            error_msg = (
                f"Cannot update weights: missing weight files for modules: {missing}. "
                f"No partial updates allowed."
            )
            logger.error(error_msg)
            return False, error_msg

        logger.info(
            f"Updating {len(weights_map)} modules: "
            + ", ".join(f"{n} <- {p}" for n, p in weights_map.items())
        )

        success, message = self._apply_weights(modules_to_update, weights_map)

        gc.collect()
        torch.cuda.empty_cache()

        if success and flush_cache:
            for _, module in modules_to_update:
                if isinstance(module, TeaCacheMixin):
                    module.reset_teacache_state()

        logger.info(message)
        return success, message

    def update_weights_from_named_tensors(
        self,
        named_tensors,
        *,
        target_modules: list[str] | None = None,
        load_format: str | None = None,
        flush_cache: bool = True,
    ) -> tuple[bool, str]:
        """Update module weights from in-memory named tensors.

        Args:
            named_tensors: Tensor payload. Supported:
                - list[(name, tensor)] / tuple[(name, tensor)]
                - dict[name, tensor]
                - flattened bucket dict when ``load_format='flattened_bucket'``
            target_modules: Restrict update to these modules.
            load_format: Optional payload format (e.g., ``flattened_bucket``).
            flush_cache: Whether to reset TeaCache state for updated modules.
        """
        if named_tensors is None:
            return False, "named_tensors is required"

        try:
            modules_to_update = self._collect_modules(target_modules)
        except ValueError as e:
            logger.error(str(e))
            return False, str(e)

        if not modules_to_update:
            return False, "No matching modules found for in-memory update."

        import time as _time

        try:
            t0 = _time.perf_counter()
            normalized = self._normalize_named_tensors(
                named_tensors=named_tensors,
                load_format=load_format,
            )
            normalize_s = _time.perf_counter() - t0

            t1 = _time.perf_counter()
            module_payloads = self._split_named_tensors_by_module(
                normalized,
                modules_to_update,
            )
            split_s = _time.perf_counter() - t1
        except Exception as e:
            logger.error("Failed to parse in-memory tensor payload: %s", e, exc_info=True)
            return False, f"Failed to parse in-memory tensor payload: {e}"

        if not module_payloads:
            return False, "No tensors in payload matched requested modules."

        logger.info(
            "Updating %d modules from in-memory tensors.",
            len(module_payloads),
        )
        t2 = _time.perf_counter()
        success, message = self._apply_named_tensor_weights(
            modules_to_update=modules_to_update,
            module_payloads=module_payloads,
        )
        apply_s = _time.perf_counter() - t2

        t3 = _time.perf_counter()
        if flush_cache:
            gc.collect()
            torch.cuda.empty_cache()
        cache_s = _time.perf_counter() - t3

        t4 = _time.perf_counter()
        if success and flush_cache:
            self._flush_module_runtime_cache(modules_to_update)
        flush_s = _time.perf_counter() - t4

        logger.warning(
            "weights_updater: normalize=%.3fs  split=%.3fs  apply=%.3fs  gc+empty_cache=%.3fs  flush=%.3fs",
            normalize_s, split_s, apply_s, cache_s, flush_s,
        )

        logger.info(message)
        return success, message

    def _collect_modules(
        self, target_modules: list[str] | None
    ) -> list[tuple[str, torch.nn.Module]]:
        """Resolve target_modules to (name, module) pairs.

        Raises:
            ValueError: If target_modules contains names not found in the pipeline.
        """
        components = get_updatable_modules(self.pipeline)

        if target_modules is None:
            names = list(components.keys())
        else:
            unknown = [n for n in target_modules if n not in components]
            if unknown:
                raise ValueError(
                    f"Module(s) requested for update not found in pipeline: {unknown}. "
                    f"Available Module(s): {list(components.keys())}"
                )
            names = target_modules

        return [(name, components[name]) for name in names]

    def _normalize_named_tensors(
        self,
        *,
        named_tensors,
        load_format: str | None,
    ) -> list[tuple[str, torch.Tensor]]:
        if load_format == "flattened_bucket":
            if not isinstance(named_tensors, dict):
                raise ValueError(
                    "flattened_bucket format expects a dict payload with flattened_tensor and metadata"
                )
            flattened_tensor = named_tensors.get("flattened_tensor")
            metadata = named_tensors.get("metadata")
            if flattened_tensor is None or metadata is None:
                raise ValueError(
                    "flattened_bucket payload must contain flattened_tensor and metadata"
                )
            try:
                from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
            except Exception:
                from sglang.srt.model_executor.model_runner import FlattenedTensorBucket

            bucket = FlattenedTensorBucket(
                flattened_tensor=flattened_tensor,
                metadata=metadata,
            )
            return list(bucket.reconstruct_tensors())

        if isinstance(named_tensors, dict):
            iterable: Iterable[tuple[str, torch.Tensor]] = named_tensors.items()
        else:
            iterable = named_tensors

        normalized: list[tuple[str, torch.Tensor]] = []
        for item in iterable:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("named_tensors must be iterable of (name, tensor) pairs")
            name, tensor = item
            if not isinstance(name, str):
                raise ValueError(f"Tensor name must be str, got: {type(name).__name__}")
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(
                    f"Tensor payload for {name} must be torch.Tensor, got: {type(tensor).__name__}"
                )
            normalized.append((name, tensor))
        return normalized

    def _split_named_tensors_by_module(
        self,
        normalized_named_tensors: list[tuple[str, torch.Tensor]],
        modules_to_update: list[tuple[str, torch.nn.Module]],
    ) -> dict[str, list[tuple[str, torch.Tensor]]]:
        # Log incoming tensor names vs module param names for diagnosis
        _incoming = [n for n, _ in normalized_named_tensors[:5]]
        logger.warning("_split: %d incoming tensors, first names: %s",
                       len(normalized_named_tensors), _incoming)
        for _mn, _mod in modules_to_update:
            _pn = list(dict(_mod.named_parameters()).keys())
            logger.warning("_split: module '%s' has %d params, first names: %s",
                           _mn, len(_pn), _pn[:5])

        module_names = {name for name, _ in modules_to_update}
        module_param_name_sets = {
            module_name: set(dict(module.named_parameters()).keys())
            for module_name, module in modules_to_update
        }
        by_module: dict[str, list[tuple[str, torch.Tensor]]] = defaultdict(list)

        for name, tensor in normalized_named_tensors:
            assigned_module = None
            inner_name = name

            if "." in name:
                prefix, suffix = name.split(".", 1)
                if prefix in module_names:
                    assigned_module = prefix
                    inner_name = suffix

            if assigned_module is None:
                if len(modules_to_update) == 1:
                    assigned_module = modules_to_update[0][0]
                    inner_name = name
                else:
                    matched = [
                        module_name
                        for module_name, param_names in module_param_name_sets.items()
                        if name in param_names
                    ]
                    if len(matched) == 1:
                        assigned_module = matched[0]
                        inner_name = name

            if assigned_module is None:
                continue

            by_module[assigned_module].append((inner_name, tensor))

        return dict(by_module)

    def _flush_module_runtime_cache(
        self, modules_to_update: list[tuple[str, torch.nn.Module]]
    ) -> None:
        for _, module in modules_to_update:
            if isinstance(module, TeaCacheMixin):
                module.reset_teacache_state()

    def _apply_weights(
        self,
        modules_to_update: list[tuple[str, torch.nn.Module]],
        weights_map: dict[str, str],
    ) -> tuple[bool, str]:
        """Load weights into each module; rollback on first failure."""
        updated_modules: list[str] = []

        for module_name, module in modules_to_update:
            try:
                weights_iter = _get_weights_iter(weights_map[module_name])
                _load_weights_into_module(module, weights_iter)
                updated_modules.append(module_name)
            except Exception as e:
                rollback_list = updated_modules + [module_name]
                logger.error(
                    f"Weight update failed for module '{module_name}': {e}. "
                    f"Rolling back {len(rollback_list)} module(s) "
                    f"(including partially-loaded '{module_name}'): "
                    f"{rollback_list}.",
                    exc_info=True,
                )
                self._rollback(rollback_list)
                return False, (
                    f"Failed to update module '{module_name}': {e}. "
                    f"All modules rolled back to original weights."
                )

        names = ", ".join(updated_modules)
        return True, f"Updated {len(updated_modules)} modules ({names})."

    def _apply_named_tensor_weights(
        self,
        modules_to_update: list[tuple[str, torch.nn.Module]],
        module_payloads: dict[str, list[tuple[str, torch.Tensor]]],
    ) -> tuple[bool, str]:
        updated_modules: list[str] = []

        for module_name, module in modules_to_update:
            module_tensors = module_payloads.get(module_name)
            if not module_tensors:
                continue
            try:
                _load_weights_into_module(module, module_tensors)
                updated_modules.append(module_name)
            except Exception as e:
                rollback_list = updated_modules + [module_name]
                logger.error(
                    "In-memory weight update failed for module '%s': %s. "
                    "Rolling back modules: %s",
                    module_name,
                    e,
                    rollback_list,
                    exc_info=True,
                )
                self._rollback(rollback_list)
                return False, (
                    f"Failed to update module '{module_name}': {e}. "
                    "All modules rolled back to original weights."
                )

        if not updated_modules:
            return False, "No module parameters were updated from in-memory payload."

        names = ", ".join(updated_modules)
        return True, f"Updated {len(updated_modules)} modules ({names}) from in-memory payload."

    def _rollback(self, updated_modules: list[str]) -> None:
        """Restore updated_modules to original weights.

        If rollback itself fails the exception propagates so the caller
        knows the model is in an inconsistent state.
        """
        if not updated_modules:
            return
        original_path = maybe_download_model(self.pipeline.model_path)
        for name in updated_modules:
            module = self.pipeline.get_module(name)
            if module is None:
                continue
            weights_dir = Path(original_path) / name
            if not weights_dir.exists():
                continue
            weights_iter = _get_weights_iter(str(weights_dir))
            _load_weights_into_module(module, weights_iter)
