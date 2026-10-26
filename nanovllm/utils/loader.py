import os
import re
from glob import glob
import json
import torch
from torch import nn
from safetensors import safe_open

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

_EXPERT_RE = re.compile(r"^(.*experts\.)(\d+)(\..*)$")
_EXPERT_PART_RE = re.compile(r"^\.(gate_proj|up_proj|down_proj)(?:\.weight)?$")


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
    loaded_param_ids = set()
    for ckpt_name in sorted(keymap):
        if ckpt_name.endswith("_scale_inv"):
            continue    # consumed by dequant at load time
        # Packed experts own one contiguous parameter per projection.  Route
        # HF per-expert checkpoint tensors directly into their local slices.
        weight_name = ckpt_name
        m = _EXPERT_RE.match(weight_name)
        packed_expert_block = None
        global_expert_id = None
        expert_part = None
        if m:
            block_name = m.group(1)[:-len("experts.")].rstrip(".")
            try:
                candidate = model.get_submodule(block_name)
            except AttributeError:
                candidate = None
            part_match = _EXPERT_PART_RE.match(m.group(3))
            if candidate is not None and hasattr(candidate, "load_expert_weight") and part_match:
                packed_expert_block = candidate
                global_expert_id = int(m.group(2))
                expert_part = part_match.group(1)
                if not candidate.owns_expert(global_expert_id):
                    continue
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
        if packed_expert_block is not None:
            packed_expert_block.load_expert_weight(global_expert_id, expert_part, tensor)
            param = (packed_expert_block.down_proj if expert_part == "down_proj"
                     else packed_expert_block.gate_up_proj)
            loaded_param_ids.add(id(param))
            continue
        for k in packed_modules_mapping:
            if k in weight_name:
                v, shard_id = packed_modules_mapping[k]
                param_name = weight_name.replace(k, v)
                try:
                    param = _get_param(model, param_name)
                except AttributeError:
                    break    # expert not owned by this rank (expert parallel)
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, tensor, shard_id)
                loaded_param_ids.add(id(param))
                break
        else:
            try:
                param = _get_param(model, weight_name)
            except AttributeError:
                continue    # expert not owned by this rank
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, tensor)
            loaded_param_ids.add(id(param))
    for module in model.modules():
        validator = getattr(module, "validate_expert_weights_loaded", None)
        if validator is not None:
            validator()
    _assert_full_coverage(model, loaded_param_ids)


def _assert_full_coverage(model: nn.Module, loaded_param_ids: set[int]):
    """Every parameter owned by this rank must have been written by the loader.

    A silently skipped weight leaves torch.empty garbage that produces
    gibberish output, so fail loudly instead.
    """
    tied = set()
    lm_head = getattr(model, "lm_head", None)
    embed = getattr(getattr(model, "model", None), "embed_tokens", None)
    if lm_head is not None and embed is not None and \
            lm_head.weight.data_ptr() == embed.weight.data_ptr():
        tied.add(id(lm_head.weight))
    missing = [name for name, p in model.named_parameters()
               if id(p) not in loaded_param_ids and id(p) not in tied]
    assert not missing, (
        f"{len(missing)} parameters were never loaded, e.g. {missing[:8]}. "
        "Loader silently skipped them — checkpoint names and model names disagree."
    )
