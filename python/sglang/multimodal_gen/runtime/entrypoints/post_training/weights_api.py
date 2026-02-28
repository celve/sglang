"""Weight update API for the diffusion engine."""

from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse

from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    DestroyWeightsUpdateGroupReqInput,
    GetWeightsChecksumReqInput,
    InitWeightsUpdateGroupReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client

router = APIRouter()


def _parse_success_message(response):
    result = response.output
    if not isinstance(result, dict):
        return False, f"Invalid scheduler response payload: {type(result).__name__}"
    return bool(result.get("success", False)), str(result.get("message", "Unknown status"))


@router.post("/update_weights_from_disk")
async def update_weights_from_disk(request: Request):
    """Update model weights from disk inplace without restarting the server."""
    body = await request.json()
    model_path = body.get("model_path")
    if not model_path:
        return ORJSONResponse(
            {"success": False, "message": "model_path is required"},
            status_code=400,
        )

    req = UpdateWeightFromDiskReqInput(
        model_path=model_path,
        flush_cache=body.get("flush_cache", True),
        target_modules=body.get("target_modules"),
    )

    try:
        response = await async_scheduler_client.forward(req)
    except Exception as e:
        return ORJSONResponse(
            {"success": False, "message": str(e)},
            status_code=500,
        )

    success, message = _parse_success_message(response)
    return ORJSONResponse(
        {"success": success, "message": message},
        status_code=200 if success else 400,
    )


@router.post("/init_weights_update_group")
async def init_weights_update_group(request: Request):
    """Initialize a temporary distributed group for external weight broadcasts."""
    body = await request.json()
    required_fields = ("master_address", "master_port", "rank_offset", "world_size")
    missing = [field for field in required_fields if body.get(field) is None]
    if missing:
        return ORJSONResponse(
            {"success": False, "message": f"Missing required fields: {missing}"},
            status_code=400,
        )

    req = InitWeightsUpdateGroupReqInput(
        master_address=body.get("master_address"),
        master_port=int(body.get("master_port")),
        rank_offset=int(body.get("rank_offset")),
        world_size=int(body.get("world_size")),
        group_name=body.get("group_name", "weight_update_group"),
        backend=body.get("backend", "nccl"),
    )

    try:
        response = await async_scheduler_client.forward(req)
    except Exception as e:
        return ORJSONResponse(
            {"success": False, "message": str(e)},
            status_code=500,
        )

    success, message = _parse_success_message(response)
    return ORJSONResponse(
        {"success": success, "message": message},
        status_code=200 if success else 400,
    )


@router.post("/destroy_weights_update_group")
async def destroy_weights_update_group(request: Request):
    """Destroy a temporary distributed group used for external weight sync."""
    body = await request.json()
    req = DestroyWeightsUpdateGroupReqInput(
        group_name=body.get("group_name", "weight_update_group"),
    )

    try:
        response = await async_scheduler_client.forward(req)
    except Exception as e:
        return ORJSONResponse(
            {"success": False, "message": str(e)},
            status_code=500,
        )

    success, message = _parse_success_message(response)
    return ORJSONResponse(
        {"success": success, "message": message},
        status_code=200 if success else 400,
    )


@router.post("/update_weights_from_tensor")
async def update_weights_from_tensor(request: Request):
    """Update model weights from serialized tensor payloads."""
    body = await request.json()
    serialized_named_tensors = body.get("serialized_named_tensors")
    if not serialized_named_tensors:
        return ORJSONResponse(
            {"success": False, "message": "serialized_named_tensors is required"},
            status_code=400,
        )

    req = UpdateWeightsFromTensorReqInput(
        serialized_named_tensors=serialized_named_tensors,
        target_modules=body.get("target_modules"),
        load_format=body.get("load_format"),
        flush_cache=body.get("flush_cache", True),
    )

    try:
        response = await async_scheduler_client.forward(req)
    except Exception as e:
        return ORJSONResponse(
            {"success": False, "message": str(e)},
            status_code=500,
        )

    success, message = _parse_success_message(response)
    return ORJSONResponse(
        {"success": success, "message": message},
        status_code=200 if success else 400,
    )


@router.post("/update_weights_from_distributed")
async def update_weights_from_distributed(request: Request):
    """Update model weights from an external distributed broadcast."""
    body = await request.json()
    names = body.get("names")
    dtypes = body.get("dtypes")
    shapes = body.get("shapes")
    if not names or not dtypes or not shapes:
        return ORJSONResponse(
            {"success": False, "message": "names, dtypes and shapes are required"},
            status_code=400,
        )

    req = UpdateWeightsFromDistributedReqInput(
        names=names,
        dtypes=dtypes,
        shapes=shapes,
        group_name=body.get("group_name", "weight_update_group"),
        target_modules=body.get("target_modules"),
        flush_cache=body.get("flush_cache", True),
    )

    try:
        response = await async_scheduler_client.forward(req)
    except Exception as e:
        return ORJSONResponse(
            {"success": False, "message": str(e)},
            status_code=500,
        )

    success, message = _parse_success_message(response)
    return ORJSONResponse(
        {"success": success, "message": message},
        status_code=200 if success else 400,
    )


@router.post("/get_weights_checksum")
async def get_weights_checksum(request: Request):
    """Return SHA-256 checksum of each requested module's weights."""
    body = await request.json()
    req = GetWeightsChecksumReqInput(
        module_names=body.get("module_names"),
    )

    try:
        response = await async_scheduler_client.forward(req)
    except Exception as e:
        return ORJSONResponse({"error": str(e)}, status_code=500)

    return ORJSONResponse(response.output, status_code=200)
