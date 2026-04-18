# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Timestep preparation stages for diffusion pipelines.

This module contains implementations of timestep preparation stages for diffusion pipelines.
"""

import inspect
from typing import Any, Callable, Tuple

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class TimestepPreparationStage(PipelineStage):
    """
    Stage for preparing timesteps for the diffusion process.

    This stage handles the preparation of the timestep sequence that will be used
    during the diffusion process.
    """

    def __init__(
        self,
        scheduler,
        prepare_extra_set_timesteps_kwargs: list[
            Callable[[Req, ServerArgs], Tuple[str, Any]]
        ] = [],
    ) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.prepare_extra_set_timesteps_kwargs = (
            prepare_extra_set_timesteps_kwargs or []
        )

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.REPLICATED

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Prepare timesteps for the diffusion process.



        Returns:
            The batch with prepared timesteps.
        """
        scheduler = self.scheduler
        device = get_local_torch_device()
        num_inference_steps = batch.num_inference_steps
        timesteps = batch.timesteps
        sigmas = batch.sigmas
        n_tokens = batch.n_tokens

        sigmas = server_args.pipeline_config.prepare_sigmas(sigmas, num_inference_steps)
        batch.sigmas = sigmas

        # Prepare extra kwargs for set_timesteps
        extra_set_timesteps_kwargs = {}
        if (
            n_tokens is not None
            and "n_tokens" in inspect.signature(scheduler.set_timesteps).parameters
        ):
            extra_set_timesteps_kwargs["n_tokens"] = n_tokens

        for callee in self.prepare_extra_set_timesteps_kwargs:
            key, value = callee(batch, server_args)
            assert isinstance(key, str)
            extra_set_timesteps_kwargs[key] = value
            if key == "mu":
                batch.extra["mu"] = value

        # Handle custom timesteps and sigmas.  The two may be supplied
        # together when the caller (e.g. DiffusionRL) wants to be the single
        # source of truth for both units and therefore bypass the scheduler's
        # internal ``timesteps = sigmas * num_train_timesteps`` derivation.
        scheduler_params = inspect.signature(scheduler.set_timesteps).parameters
        accepts_timesteps = "timesteps" in scheduler_params
        accepts_sigmas = "sigmas" in scheduler_params

        set_timesteps_kwargs: dict[str, Any] = {
            "device": device,
            **extra_set_timesteps_kwargs,
        }

        if timesteps is not None and not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        if sigmas is not None and not accepts_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )

        if timesteps is not None and sigmas is not None:
            if len(timesteps) != len(sigmas):
                raise ValueError(
                    "Custom `timesteps` and `sigmas` must have matching lengths; "
                    f"got len(timesteps)={len(timesteps)}, len(sigmas)={len(sigmas)}."
                )
            set_timesteps_kwargs["timesteps"] = timesteps
            set_timesteps_kwargs["sigmas"] = sigmas
        elif timesteps is not None:
            set_timesteps_kwargs["timesteps"] = timesteps
        elif sigmas is not None:
            set_timesteps_kwargs["sigmas"] = sigmas
        else:
            set_timesteps_kwargs["num_inference_steps"] = num_inference_steps

        scheduler.set_timesteps(**set_timesteps_kwargs)
        timesteps = scheduler.timesteps

        # Update batch with prepared timesteps
        batch.timesteps = timesteps
        if not batch.is_warmup:
            self.log_debug("timesteps: %s", timesteps)
        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify timestep preparation stage inputs."""
        result = VerificationResult()
        result.add_check(
            "num_inference_steps", batch.num_inference_steps, V.positive_int
        )
        result.add_check("timesteps", batch.timesteps, V.none_or_tensor)
        result.add_check("sigmas", batch.sigmas, V.none_or_list)
        result.add_check("n_tokens", batch.n_tokens, V.none_or_positive_int)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify timestep preparation stage outputs."""
        if (
            batch.is_warmup
            and isinstance(batch.timesteps, torch.Tensor)
            and torch.isnan(batch.timesteps).any()
        ):
            # when num-inference-steps == 1, the last sigma being 1, the 1 / last_sigma could be nan
            # this a workaround for warmup req only
            batch.timesteps = torch.ones(
                (1,), dtype=torch.float32, device=get_local_torch_device()
            )

        result = VerificationResult()
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.with_dims(1)])
        return result
