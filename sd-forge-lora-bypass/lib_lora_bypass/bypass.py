# LoRA Bypass for Forge Neo - core.
#
# Forge's "Diffusion in Low Bits = ... (fp16 LoRA)" mode ("on-the-fly") registers every LoRA as an OnlineLoRAPatch in
# ModelPatcher.weight_wrapper_patches; on every call of a patched layer Forge dequantizes the whole base weight, adds
# up @ down to it and runs a dense matmul. This module changes exactly one thing: when a Linear layer whose
# weight functions are all plain LoRAs is called as a module (module(x) -> forward), it runs the layer's own fast path
# WITHOUT the weight functions and adds the low-rank branch on the output:
#
#     y = W x (+ b)  +  sum_i  strength_i * alpha_i / rank_i * up_i (down_i x)
#
# Everything else stays Forge's on-the-fly: LoRA registration is untouched (OnlineLoRAPatch objects stay in
# weight_wrapper_patches and in module.weight_function), so fused kernels that read the weight directly
# (linear_input_act), unsupported adapters (LoKr, LoHa, DoRA, ...), convolutions, bias patches, LoRA Control and
# Forge's memory reservation keep working exactly as before.
#
# The replacement forward is installed with Forge's own object patches (ModelPatcher.add_object_patch): Forge applies
# them in patch_model(), removes them in unpatch_model() and copies them in clone(), so this module needs no lifecycle
# code of its own and never changes a module permanently.

import logging
import threading
import weakref
from collections import Counter

import torch
import torch.nn.functional as F

TAG = "[LoRA Bypass]"
logger = logging.getLogger("lora_bypass")

_ORIG_ADD_PATCHES_ATTR = "__lora_bypass_original__"


class _State:
    """Process-wide switches. `enabled` mirrors the Settings option; it is refreshed once per generation
    (Script.process) and by the option's onchange callback, never read from shared.opts per layer."""

    def __init__(self):
        self.enabled = True
        self.force = None  # None | "otf" | "bp" (self-test only)
        self.installed = False
        self.lock = threading.Lock()


STATE = _State()


class _Stats:
    """Per-generation runtime counters (cheap: one set.add and one int add per bypassed call)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.bypass_calls = 0
        self.fallback_calls = 0
        self.bypass_modules = set()
        self.fallback_modules = {}

    def snapshot(self):
        return {
            "bypass_calls": self.bypass_calls,
            "fallback_calls": self.fallback_calls,
            "bypass_modules": len(self.bypass_modules),
            "fallback_modules": len(self.fallback_modules),
            "fallback_reasons": dict(Counter(self.fallback_modules.values())),
        }


STATS = _Stats()


def online_lora_mode() -> bool:
    from backend.args import dynamic_args

    return bool(getattr(dynamic_args, "online_lora", False))


def bypass_active() -> bool:
    if STATE.force is not None:
        return STATE.force == "bp"
    return STATE.enabled and online_lora_mode()


# ---------------------------------------------------------------------------------------------------------------------
# Classification


def _lora_adapter_cls():
    try:
        from modules_forge.packages.comfy.weight_adapter.lora import LoRAAdapter

        return LoRAAdapter
    except Exception:  # pragma: no cover - layout of a future Forge
        return None


def _online_patch_cls():
    from backend.patcher.base import OnlineLoRAPatch

    return OnlineLoRAPatch


def is_linear_module(m) -> bool:
    """A Forge Linear op (plain, mixed-precision int8/fp8/nvfp4, fp8-fast, GGUF): has Forge weight functions and a
    `Linear` class name. Embeddings / convolutions are excluded even if their weight is 2-D."""
    if not hasattr(m, "weight_function") or not hasattr(m, "bias_function"):
        return False
    if isinstance(m, torch.nn.Linear):
        return True
    return type(m).__name__ == "Linear"


def classify_patch(fn):
    """None if this weight function can run as a bypass branch, else a short reason."""
    if not isinstance(fn, _online_patch_cls()):
        return "other weight function"
    try:
        strength, v, strength_model, offset, function = fn.patch[0]
    except Exception:
        return "unknown patch layout"
    if function is not None:
        return "patch function"
    if strength_model != 1.0:
        return "strength_model"
    lora_cls = _lora_adapter_cls()
    if lora_cls is None or not isinstance(v, lora_cls):
        name = getattr(v, "name", None)
        if isinstance(name, str):
            return name  # lokr / loha / glora / oft / boft / ...
        return "diff" if isinstance(v, (tuple, list)) else "unsupported adapter"
    try:
        up, down, alpha, mid, dora_scale, reshape = v.weights
    except Exception:
        return "unknown lora layout"
    if dora_scale is not None:
        return "dora"
    if mid is not None:
        return "lora mid (locon)"
    if reshape is not None:
        return "reshape"
    if not (isinstance(up, torch.Tensor) and isinstance(down, torch.Tensor)) or up.ndim != 2 or down.ndim != 2:
        return "conv lora"
    if up.shape[1] != down.shape[0]:
        return "rank mismatch"
    if offset is not None and (len(offset) < 3 or offset[0] != 0):
        return "offset axis"
    return None


def _known_dims(m):
    """(in, out) of a Linear if they are known without touching quantized storage, else None entries."""
    fin = getattr(m, "in_features", None)
    fout = getattr(m, "out_features", None)
    return (fin if isinstance(fin, int) else None), (fout if isinstance(fout, int) else None)


class _Branch:
    __slots__ = ("fn", "up", "down", "alpha_r", "start", "length")

    def __init__(self, fn, up, down, alpha_r, start, length):
        self.fn = fn  # OnlineLoRAPatch: strength is read from fn.patch[0][0] on every call (LoRA Control)
        self.up = up
        self.down = down
        self.alpha_r = alpha_r
        self.start = start
        self.length = length


class _Plan:
    __slots__ = ("wf", "wf_items", "bf", "bf_len", "branches", "in_features", "full_out", "min_out", "reason")

    def output_ok(self, out) -> bool:
        if not isinstance(out, torch.Tensor):
            return False
        n = out.shape[-1]
        return (self.full_out is None or n == self.full_out) and n >= self.min_out

    def matches(self, wf, bf):
        if wf is not self.wf or bf is not self.bf or len(wf) != len(self.wf_items) or len(bf) != self.bf_len:
            return False
        for a, b in zip(wf, self.wf_items):
            if a is not b:
                return False
        return True


def build_plan(m, wf, bf) -> "_Plan":
    plan = _Plan()
    plan.wf, plan.wf_items, plan.bf, plan.bf_len = wf, tuple(wf), bf, len(bf)
    plan.branches = None
    plan.in_features = None
    plan.full_out = None
    plan.min_out = 0
    plan.reason = None

    if not is_linear_module(m):
        plan.reason = "not linear"
        return plan
    if len(bf) > 0:
        plan.reason = "bias patch"
        return plan
    if len(wf) == 0:
        plan.reason = "no lora"
        return plan
    branches = []
    fin, fout = _known_dims(m)
    for fn in wf:
        reason = classify_patch(fn)
        if reason is not None:
            plan.reason = reason
            return plan
        _, v, _, offset, _ = fn.patch[0]
        up, down, alpha = v.weights[0], v.weights[1], v.weights[2]
        rank = down.shape[0]
        alpha_r = (float(alpha) / rank) if alpha is not None else 1.0
        if offset is None:
            start, length = None, up.shape[0]
            if (fout is not None and length != fout) or (plan.full_out is not None and length != plan.full_out):
                plan.reason = "shape"
                return plan
            plan.full_out = length
        else:
            start, length = int(offset[1]), int(offset[2])
            if up.shape[0] != length or start < 0 or (fout is not None and start + length > fout):
                plan.reason = "shape"
                return plan
        plan.min_out = max(plan.min_out, (start or 0) + length)
        if plan.in_features is None:
            plan.in_features = down.shape[1]
        elif plan.in_features != down.shape[1]:
            plan.reason = "shape"
            return plan
        branches.append(_Branch(fn, up, down, alpha_r, start, length))
    if fin is not None and plan.in_features != fin:
        plan.reason = "shape"
        return plan
    plan.branches = branches
    return plan


# ---------------------------------------------------------------------------------------------------------------------
# The replacement forward

_cast_to_device = None


def _to(t, x):
    """LoRA matrix -> device/dtype of the activations. Copied on every call like Forge's on-the-fly does (same memory
    profile: the LoRA stays in system RAM and Forge's own reservation for it is untouched)."""
    global _cast_to_device
    if _cast_to_device is None:
        from backend.memory_management import cast_to_device

        _cast_to_device = cast_to_device
    return _cast_to_device(t, x.device, x.dtype)


class BypassForward:
    """Object patch for `<module path>.forward` of one LoRA'd Linear. Holds the module by weakref; the computation
    plan is cached on this object and keyed on the identity of the module's weight_function list (Forge rebuilds
    that list on every load), so nothing is ever stored on the shared module."""

    __slots__ = ("_ref", "path", "_plan", "__weakref__")

    def __init__(self, module, path):
        self._ref = weakref.ref(module)
        self.path = path
        self._plan = None

    def _get_plan(self, m):
        wf = m.weight_function
        bf = m.bias_function
        plan = self._plan
        if plan is None or not plan.matches(wf, bf):
            plan = build_plan(m, wf, bf)
            self._plan = plan
        return plan

    def __call__(self, x, *args, **kwargs):
        m = self._ref()
        if m is None:
            raise RuntimeError(f"{TAG} module {self.path} no longer exists")
        forward = type(m).forward  # never a saved bound method: object-patch backups are shared by clones
        if not bypass_active():
            return forward(m, x, *args, **kwargs)
        plan = self._get_plan(m)
        if (
            plan.branches is None
            or type(x) is not torch.Tensor
            or not x.is_floating_point()
            or x.shape[-1] != plan.in_features
        ):
            STATS.fallback_calls += 1
            STATS.fallback_modules[self.path] = plan.reason or "input"
            return forward(m, x, *args, **kwargs)

        wf = m.weight_function
        m.weight_function = []
        try:
            out = forward(m, x, *args, **kwargs)
        finally:
            m.weight_function = wf

        if not plan.output_ok(out):
            # unexpected output layout: never guess, recompute this call on Forge's on-the-fly path (sticky)
            plan.branches = None
            plan.reason = "output shape"
            STATS.fallback_calls += 1
            STATS.fallback_modules[self.path] = plan.reason
            return forward(m, x, *args, **kwargs)

        for br in plan.branches:
            strength = br.fn.patch[0][0]
            if strength == 0.0:
                continue
            # computed AFTER the base forward: some Forge kernels clamp x in place (fp8_linear), and on-the-fly
            # multiplies (W + dW) with that same clamped x
            z = F.linear(x, _to(br.down, x))
            scale = strength * br.alpha_r
            if scale != 1.0:
                z.mul_(scale)
            d = F.linear(z, _to(br.up, x))
            if d.dtype != out.dtype:
                d = d.to(out.dtype)
            if br.start is None:
                out.add_(d)
            else:
                out.narrow(-1, br.start, br.length).add_(d)
        STATS.bypass_calls += 1
        STATS.bypass_modules.add(self.path)
        return out

    # self-test helpers ------------------------------------------------------------------------------------------
    def module(self):
        return self._ref()

    def plan(self):
        m = self._ref()
        return None if m is None else self._get_plan(m)


# ---------------------------------------------------------------------------------------------------------------------
# Registration: wrap ModelPatcher.add_patches


def _register(patcher, keys):
    from backend import utils

    for k in keys:
        key = k if isinstance(k, str) else k[0]
        if not isinstance(key, str) or not key.endswith(".weight"):
            continue
        path = key[: -len(".weight")]
        name = path + ".forward"
        existing = patcher.object_patches.get(name)
        if isinstance(existing, BypassForward):
            continue
        if existing is not None:
            continue  # someone else patches this forward: leave the layer on Forge's path
        try:
            module = utils.get_attr(patcher.model, path)
        except Exception:
            continue
        if not is_linear_module(module):
            continue
        patcher.add_object_patch(name, BypassForward(module, path))


def install():
    """Idempotent: wraps backend.patcher.base.ModelPatcher.add_patches once per process."""
    with STATE.lock:
        if STATE.installed:
            return True
        from backend.patcher.base import ModelPatcher

        current = ModelPatcher.add_patches
        original = getattr(current, _ORIG_ADD_PATCHES_ATTR, current)

        def add_patches(self, *args, **kwargs):
            keys = original(self, *args, **kwargs)
            if kwargs.get("online_mode"):
                try:
                    _register(self, keys)
                except Exception as e:  # never break LoRA loading: the layer simply stays on-the-fly
                    print(f"{TAG} could not register bypass layers ({e}); LoRA stays on Forge's on-the-fly path")
            return keys

        setattr(add_patches, _ORIG_ADD_PATCHES_ATTR, original)
        add_patches.__wrapped__ = original
        ModelPatcher.add_patches = add_patches
        STATE.installed = True
        return True


# ---------------------------------------------------------------------------------------------------------------------
# Static summary of a patcher (printed once after LoRA loading, used for the infotext)


def summarize_patcher(patcher):
    """Counts LoRA'd weights of one ModelPatcher: (bypassable, total, Counter(reasons))."""
    from backend import utils

    total = 0
    ok = 0
    reasons = Counter()
    wwp = getattr(patcher, "weight_wrapper_patches", None) or {}
    bias_keys = {k for k in wwp if k.endswith(".bias")}
    for key, fns in wwp.items():
        if not key.endswith(".weight"):
            continue
        total += 1
        path = key[: -len(".weight")]
        if not isinstance(patcher.object_patches.get(path + ".forward"), BypassForward):
            try:
                module = utils.get_attr(patcher.model, path)
                reasons["conv / other layer" if not is_linear_module(module) else "forward patched by another extension"] += 1
            except Exception:
                reasons["missing module"] += 1
            continue
        if path + ".bias" in bias_keys:
            reasons["bias patch"] += 1
            continue
        reason = None
        for fn in fns:
            reason = classify_patch(fn)
            if reason is not None:
                break
        if reason is None:
            ok += 1
        else:
            reasons[reason] += 1
    return ok, total, reasons


def format_summary(parts):
    """parts: list of (label, ok, total, reasons)."""
    chunks = []
    for label, ok, total, reasons in parts:
        if total == 0:
            continue
        s = f"{label}: {ok}/{total} LoRA layers bypassed"
        fb = total - ok
        if fb:
            s += f", {fb} fall back to on-the-fly (" + ", ".join(f"{r}: {n}" for r, n in reasons.most_common()) + ")"
        chunks.append(s)
    return "; ".join(chunks)
