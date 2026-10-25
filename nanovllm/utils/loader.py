import os
from glob import glob
import json
import torch
from torch import nn
from safetensors import safe_open

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


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


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    block = _fp8_block_size(path)
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            keys = set(f.keys())
            for weight_name in sorted(f.keys()):
                if weight_name.endswith("_scale_inv"):
                    continue    # consumed by dequant at load time
                tensor = f.get_tensor(weight_name)
                if block is not None and tensor.dtype in FP8_DTYPES:
                    scale_name = weight_name + "_scale_inv"
                    assert scale_name in keys, f"missing {scale_name}"
                    if tensor.dtype == torch.uint8:
                        tensor = tensor.view(torch.float8_e4m3fn)
                    tensor = dequant_fp8_block(tensor, f.get_tensor(scale_name), block)
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        try:
                            param = model.get_parameter(param_name)
                        except AttributeError:
                            continue    # expert not owned by this rank (expert parallel)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, tensor, shard_id)
                        break
                else:
                    try:
                        param = model.get_parameter(weight_name)
                    except AttributeError:
                        continue    # expert not owned by this rank
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, tensor)
