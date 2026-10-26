"""Zero-copy access to the installed vllm-musa Triton MoE.

``qwen3_moe.py`` stores experts packed as ``[E, in, out]`` because that is what
``torch._grouped_mm`` consumes.  vLLM's ``fused_experts_impl`` asserts
``w1.stride(-1) == 1`` (its upstream callers always pass ``[E, out, in]``
copies), but the Triton kernel itself takes every stride as an argument and is
correct for the transposed view -- verified numerically and under graph capture
in ``tests/p6_vendor_zero_copy.py`` (max_rel ~4e-3 vs an fp32 reference, 0.0 MiB
extra memory, replay identical).

Copying the 30B experts into vLLM's layout costs 384 + 192 MiB per layer, i.e.
27 GiB for 48 layers, which cannot fit next to the 28.6 GiB of weights on a
48 GB card.  That copy -- not a vendor workspace -- is what OOM'd every P6
capture attempt (``tests/p6_moe_memory_probe.py``: 0.0 MiB per vendor call,
576 MiB per layer for the copy).

Instead of pinning the whole engine to ``python -O`` (which would also disable
unrelated safety asserts), rebind a source-level copy of ``fused_experts_impl``
with exactly those two layout asserts removed.  Nothing on disk is modified and
the numeric result is gated by the probe above.
"""
import inspect

import vllm.model_executor.layers.fused_moe.fused_moe as _vendor

_PATCH_MARKER = "_nanovllm_zero_copy"


def _install_patch():
    if getattr(_vendor, _PATCH_MARKER, False):
        return
    source = inspect.getsource(_vendor.fused_experts_impl)
    patched = source.replace(
        '    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"\n'
        '    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"\n',
        "    # nanovllm: [E, in, out] experts arrive as [E, out, in] views;\n"
        "    # the kernel reads explicit strides (validated in tests/p6_vendor_zero_copy.py).\n",
    )
    if patched == source:
        raise RuntimeError(
            "vendor fused_experts_impl no longer matches the patched source; "
            "re-check the layout asserts before using the zero-copy path"
        )
    exec(compile(patched, _vendor.__file__, "exec"), _vendor.__dict__)
    setattr(_vendor, _PATCH_MARKER, True)


def fused_experts_zero_copy(
    hidden_states,
    gate_up_proj,
    down_proj,
    topk_weights,
    topk_ids,
    global_num_experts,
    expert_map,
    use_mp22=False,
):
    """Run the vendor Triton MoE on packed ``[E, in, out]`` expert weights.

    ``gate_up_proj`` is ``[E, hidden, 2*inter]`` and ``down_proj`` is
    ``[E, inter, hidden]``; both are passed as transposed views, so no copy is
    made and the graph pool sees stable storage addresses.

    ``use_mp22`` selects the vendor's S4000 fast path.  It is off by default
    because it aborted at 1024 tokens in earlier tests; decode-size benchmarks
    opt in explicitly.
    """
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    _install_patch()
    _vendor.use_mp22_fused_moe = bool(use_mp22)
    return _vendor.fused_experts(
        hidden_states,
        gate_up_proj.transpose(1, 2),
        down_proj.transpose(1, 2),
        topk_weights.float(),
        topk_ids.int(),
        activation=MoEActivation.SILU,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
    )
