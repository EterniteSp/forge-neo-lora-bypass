# LoRA Bypass - numeric self-test (GET /lora_bypass/selftest, only with --api).
#
# On the currently loaded model with its current LoRAs, feeds random inputs to up to `n` LoRA'd Linear layers and
# compares, per layer:
#   bp   - this extension's bypass forward
#   otf  - Forge's on-the-fly forward (full weight_function)
#   base - the layer's own fast path without LoRA (what bypass builds on)
#   ref  - float32 reference  x @ (W_deq + dW)^T + b   with dW = sum strength * alpha / rank * up @ down
# Layers that cannot be bypassed must give exactly the on-the-fly result (torch.equal).

import math

import torch
import torch.nn.functional as F

from . import bypass as bp_core


def _rel(a, b):
    a = a.float()
    b = b.float()
    den = b.norm().item()
    return (a - b).norm().item() / den if den > 0 else float("inf")


def _dequantized_weight_and_bias(m, x):
    """W and b exactly as Forge's on-the-fly sees them before adding the LoRA (compute dtype, on x.device)."""
    from backend.operations import weights_manual_cast

    saved_wf, saved_bf = m.weight_function, m.bias_function
    m.weight_function, m.bias_function = [lambda w: w], []
    try:
        kw = {}
        if "GGUF" in type(m).__qualname__:
            from backend.operations_gguf import dequantize_tensor

            kw = dict(weight_fn=dequantize_tensor, skip_bias_dtype=True)
        w, b, _ = weights_manual_cast(m, x, **kw)
    finally:
        m.weight_function, m.bias_function = saved_wf, saved_bf
    if b is not None:
        b = b.to(device=x.device)
    return w.to(device=x.device), b


def _delta(plan, out_features, device):
    dw = torch.zeros(out_features, plan.in_features, dtype=torch.float32, device=device)
    for br in plan.branches:
        s = br.fn.patch[0][0] * br.alpha_r
        d = s * (br.up.to(device=device, dtype=torch.float32) @ br.down.to(device=device, dtype=torch.float32))
        if br.start is None:
            dw += d
        else:
            dw[br.start : br.start + br.length] += d
    return dw


def _run(bf, m, x, mode):
    bp_core.STATE.force = mode
    try:
        return bf(x)
    finally:
        bp_core.STATE.force = None


def _base(m, x):
    wf = m.weight_function
    m.weight_function = []
    try:
        return type(m).forward(m, x)
    finally:
        m.weight_function = wf


def _pick(items, n):
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


@torch.inference_mode()
def run(patcher, n=64, tokens=256, seed=0, label="unet", dtype=None):
    from backend import memory_management, utils

    memory_management.load_models_gpu([patcher])
    device = memory_management.get_torch_device()
    if dtype is None:
        dtype = getattr(patcher.model, "computation_dtype", None) or torch.bfloat16

    eligible, fallback, skipped = [], [], []
    for key in patcher.weight_wrapper_patches:
        if not key.endswith(".weight"):
            continue
        path = key[: -len(".weight")]
        bf = patcher.object_patches.get(path + ".forward")
        if not isinstance(bf, bp_core.BypassForward):
            skipped.append(path)
            continue
        m = bf.module()
        plan = bf.plan()
        (eligible if plan.branches is not None else fallback).append((path, bf, m, plan))

    gen = torch.Generator(device="cpu").manual_seed(seed)
    rows, failures, nonfinite = [], [], []
    s = bp_core.STATS
    stats_before = {"bypass_calls": s.bypass_calls, "fallback_calls": s.fallback_calls,
                    "bypass_modules": set(s.bypass_modules), "fallback_modules": dict(s.fallback_modules)}

    def make_x(in_features):
        return torch.randn(tokens, in_features, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)

    for path, bf, m, plan in _pick(eligible, n):
        x = make_x(plan.in_features)
        y_bp = _run(bf, m, x, "bp")
        y_otf = _run(bf, m, x, "otf")
        y_base = _base(m, x)
        w, b = _dequantized_weight_and_bias(m, x)
        dw = _delta(plan, w.shape[0], device)
        xf = x.float()
        bf32 = b.float() if b is not None else None
        ref_nolora = F.linear(xf, w.float(), bf32)
        ref_delta = F.linear(xf, dw)
        ref = ref_nolora + ref_delta
        if not bool(torch.isfinite(ref).all()):
            # the model's own weights overflow for this input (e.g. an unused, unscaled last CLIP layer):
            # no meaningful error; record whether bypass and on-the-fly agree on where the non-finite values are
            nonfinite.append({"layer": path, "same_nonfinite_pattern_as_otf":
                              bool(torch.equal(torch.isfinite(y_bp), torch.isfinite(y_otf)))})
            continue
        y_otf_nolora = F.linear(x, w.to(x.dtype), b.to(x.dtype) if b is not None else None)  # on-the-fly math, no LoRA
        r = {
            "layer": path,
            "class": type(m).__qualname__,
            "weight": type(m.weight).__name__ if m.weight is not None else None,
            "err_bp": _rel(y_bp, ref),
            "err_otf": _rel(y_otf, ref),
            "err_base_path": _rel(y_base, ref_nolora),
            "err_bp_lora": _rel(y_bp.float() - y_base.float(), ref_delta),
            "err_otf_lora": _rel(y_otf.float() - y_otf_nolora.float(), ref_delta),
            "lora_share": ref_delta.norm().item() / max(ref.norm().item(), 1e-30),
        }
        # bypass may not add error beyond the layer's own fast path (same as without LoRA), and its LoRA term
        # must be at least as precise as on-the-fly's
        total_ok = r["err_bp"] <= 1.5 * r["err_otf"] + 1e-3 or r["err_bp"] <= 1.1 * r["err_base_path"] + 1e-3
        lora_ok = r["err_bp_lora"] <= 1.5 * r["err_otf_lora"] + 1e-3
        r["pass"] = bool(total_ok and lora_ok and math.isfinite(r["err_bp"]))
        rows.append(r)
        if not r["pass"]:
            failures.append(path)

    fb_rows = []
    for path, bf, m, plan in _pick(fallback, n):
        in_f = getattr(m, "in_features", None)
        if not isinstance(in_f, int):
            continue
        x = make_x(in_f)
        same = torch.equal(_run(bf, m, x, "bp"), _run(bf, m, x, "otf"))
        fb_rows.append({"layer": path, "reason": plan.reason, "equal_to_otf": bool(same)})
        if not same:
            failures.append(path)

    bp_core.STATS.__dict__.update(stats_before)  # self-test calls are not generation statistics

    def agg(key):
        v = sorted(r[key] for r in rows)
        return {"max": v[-1], "median": v[len(v) // 2]} if v else None

    reasons = {}
    for _, _, _, plan in fallback:
        reasons[plan.reason] = reasons.get(plan.reason, 0) + 1
    return {
        "target": label,
        "compute_dtype": str(dtype),
        "tokens": tokens,
        "lora_layers": len(eligible) + len(fallback) + len(skipped),
        "bypass_layers": len(eligible),
        "fallback_layers": len(fallback) + len(skipped),
        "fallback_reasons": {**reasons, **({"conv / other layer": len(skipped)} if skipped else {})},
        "tested_bypass": len(rows),
        "tested_fallback": len(fb_rows),
        "skipped_nonfinite_reference": nonfinite,
        "err_bp": agg("err_bp"),
        "err_otf": agg("err_otf"),
        "err_base_path": agg("err_base_path"),
        "err_bp_lora": agg("err_bp_lora"),
        "err_otf_lora": agg("err_otf_lora"),
        "passed": len(failures) == 0,
        "failures": failures,
        "layers": rows,
        "fallback": fb_rows,
    }
