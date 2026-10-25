import os
import re
from glob import glob
import json
import torch
from torch import nn
from safetensors import safe_open

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

_EXPERT_RE = re.compile(r"^(.*experts\.)(\d+)(\..*)$")


def _expert_offset(model: nn.Module) -> int | None:
    """Global expert id of local experts.0 for expert-parallel MoE blocks."""
    for m in model.modules():
        start = getattr(m, "expert_start", None)
        if start is not None:
            return start
    return None


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def dequant_fp8_block(w_fp8: torch.Tensor, scale_inv: torch.Tensor, block: list[int]) -> torch.Tensor:
    """Qwen-style FP8 block quantization: w = fp8 * scale_inv (expanded per 128x128 block)."""
    w = w_fp8.to(torch.float32)
    rows, cols = w.shape
    rb, cb = block
    s = scale_inv.float().repeat_interleave(rb, dim=0)[:rows, :]
    s = s.repeat_interleave(cb, dim=1)[:, :cols]
    return (w * s).to(torch.bfloat16)


def _fp8_block_size(path: str) -> list[int] | None:
    cfg_file = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_file):
        return None
    with open(cfg_file) as f:
        qc = json.load(f).get("quantization_config") or {}
    if qc.get("quant_method") != "fp8":
        return None
    return qc.get("weight_block_size", [128, 128])


def _get_param(model: nn.Module, name: str):
    """Resolve a checkpoint name to a model parameter.

    Expert weights are stored HF-style (gate_proj/up_proj/down_proj.weight)
    but our experts hold bare Parameters (gate_up_proj, down_proj), so also
    try the name without the .weight suffix.
    """
    try:
        return model.get_parameter(name)
    except AttributeError:
        if name.endswith(".weight"):
            return model.get_parameter(name[: -len(".weight")])
        raise


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    block = _fp8_block_size(path)
    expert_start = _expert_offset(model)
    # a weight and its _scale_inv can live in different shards: index all files first
    keymap = {}
    for file in sorted(glob(os.path.join(path, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as f:
            for k in f.keys():
                keymap[k] = file
    for ckpt_name in sorted(keymap):
        if ckpt_name.endswith("_scale_inv"):
            continue    # consumed by dequant at load time
        # expert parallel: global expert id -> local ModuleList index
        weight_name = ckpt_name
        m = _EXPERT_RE.match(weight_name)
        if m and expert_start is not None:
            gid = int(m.group(2))
            if gid < expert_start:
                continue    # owned by another rank
            weight_name = f"{m.group(1)}{gid - expert_start}{m.group(3)}"
        with safe_open(keymap[ckpt_name], "pt", "cpu") as f:
            tensor = f.get_tensor(ckpt_name)
            if block is not None and tensor.dtype in FP8_DTYPES:
                scale_name = ckpt_name + "_scale_inv"
                assert scale_name in keymap, f"missing {scale_name}"
                with safe_open(keymap[scale_name], "pt", "cpu") as sf:
                    scale = sf.get_tensor(scale_name)
                if tensor.dtype == torch.uint8:
                    tensor = tensor.view(torch.float8_e4m3fn)
                tensor = dequant_fp8_block(tensor, scale, block)
        for k in packed_modules_mapping:
            if k in weight_name:
                v, shard_id = packed_modules_mapping[k]
                param_name = weight_name.replace(k, v)
                try:
                    param = _get_param(model, param_name)
                except AttributeError:
                    pass    # expert not owned by this rank (expert parallel)
                else:
                    weight_loader = getattr(param, "weight_loader")
                    weight_loader(param, tensor, shard_id)
                break
        else:
            try:
                param = _get_param(model, weight_name)
            except AttributeError:
                continue    # expert not owned by this rank
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, tensor)
