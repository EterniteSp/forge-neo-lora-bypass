# LoRA Bypass for Forge Neo
# In "Diffusion in Low Bits = ... (fp16 LoRA)" (and GGUF) modes, LoRA'd Linear layers keep their fast int8 / fp8 /
# fp16 kernel and get the LoRA as a separate low-rank branch: y = W x + scale * up(down(x)).
# Forge itself is not modified on disk; delete this folder and everything is exactly as before.

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from modules import script_callbacks, scripts, shared  # noqa: E402

from lib_lora_bypass import bypass  # noqa: E402

VERSION = "1.0.0"
OPTION = "lora_bypass_enabled"
OPTION_VERBOSE = "lora_bypass_verbose"
INFOTEXT_KEY = "LoRA bypass"

HELP = (
    "Computes each LoRA as a separate low-rank branch (y = W·x + scale·B(A·x)) instead of rebuilding the full weight "
    "on every step. The base layer keeps its fast int8 / fp8 kernel and the LoRA is applied at full precision, so the "
    "result does not depend on which part of the model sits in VRAM. Works only with Diffusion in Low Bits = "
    "'... (fp16 LoRA)' (and GGUF, which always uses it); in 'Automatic' (merged) mode it does nothing. Layers that "
    "cannot be bypassed (LoKr, LoHa, DoRA, convolutions, fused kernels) stay on the regular on-the-fly path. "
    "Recommended for diffusion models of about 8 GB and larger on 12 GB GPUs (12 GB+ on 16 GB, 20 GB+ on 24 GB: "
    "roughly VRAM minus 4 GB, i.e. models that don't fit entirely in VRAM) together with Diffusion in Low Bits = "
    "'... (fp16 LoRA)', to get the precision and stability of on-the-fly LoRA at close to no-LoRA speed. "
    "Takes effect from the next generation; no LoRA reload is needed. When active, the console prints one "
    "'[LoRA Bypass] ...' line after LoRAs are loaded and the image infotext gets 'LoRA bypass: on'."
)

try:
    bypass.install()
except Exception as e:  # never break Forge start-up
    print(f"{bypass.TAG} failed to install, Forge keeps its own LoRA path: {e}")


def _refresh_enabled(*_):
    bypass.STATE.enabled = bool(getattr(shared.opts, OPTION, True))


def on_ui_settings():
    section = ("lora_bypass", "LoRA Bypass")
    explanation = shared.OptionHTML(f"<b>LoRA Bypass {VERSION}</b>: {HELP}")
    explanation.section = section
    shared.opts.add_option("lora_bypass_explanation", explanation)
    shared.opts.add_option(
        OPTION,
        shared.OptionInfo(True, "Enable LoRA bypass (fp16 LoRA modes)", gr.Checkbox, {"interactive": True},
                          onchange=_refresh_enabled, section=section)
        .info("only with Diffusion in Low Bits = '... (fp16 LoRA)' or GGUF; takes effect from the next generation"),
    )
    shared.opts.add_option(
        OPTION_VERBOSE,
        shared.OptionInfo(False, "Print per-generation bypass statistics to the console", gr.Checkbox,
                          {"interactive": True}, section=section),
    )


script_callbacks.on_ui_settings(on_ui_settings)


class _Last:
    summary_key = None
    summary = ""
    active = False


def _patchers(p):
    fo = getattr(p.sd_model, "forge_objects", None)
    out = []
    if fo is None:
        return out
    unet = getattr(fo, "unet", None)
    if unet is not None:
        out.append(("UNet", unet))
    clip = getattr(fo, "clip", None)
    if clip is not None and getattr(clip, "patcher", None) is not None:
        out.append(("text encoder", clip.patcher))
    return out


class LoRABypassScript(scripts.Script):
    def title(self):
        return "LoRA Bypass"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def process(self, p, *args):
        # after override_settings are applied, before LoRAs are activated
        _refresh_enabled()
        bypass.STATS.reset()

    def process_before_every_sampling(self, p, *args, **kwargs):
        _refresh_enabled()
        if not bypass.STATE.enabled or not bypass.online_lora_mode():
            _Last.active = False
            return
        parts = []
        for label, patcher in _patchers(p):
            if not patcher.has_online_lora():
                continue
            ok, total, reasons = bypass.summarize_patcher(patcher)
            parts.append((label, ok, total, reasons))
        if not parts or sum(x[1] for x in parts) == 0:
            _Last.active = False
            return
        _Last.active = True
        key = (getattr(p.sd_model, "current_lora_hash", None), tuple((l, o, t) for l, o, t, _ in parts))
        if key != _Last.summary_key:
            _Last.summary_key = key
            _Last.summary = bypass.format_summary(parts)
            print(f"{bypass.TAG} {_Last.summary}")
        p.extra_generation_params[INFOTEXT_KEY] = "on"

    def postprocess(self, p, processed, *args):
        if getattr(shared.opts, OPTION_VERBOSE, False) and (bypass.STATS.bypass_calls or bypass.STATS.fallback_calls):
            s = bypass.STATS.snapshot()
            print(f"{bypass.TAG} this generation: {s['bypass_modules']} layers ran {s['bypass_calls']} bypass calls; "
                  f"{s['fallback_modules']} layers called through forward stayed on-the-fly {s['fallback_reasons'] or ''}")


# ---------------------------------------------------------------------------------------------------------------------
# API (only with --api)


def _status():
    s = bypass.STATS.snapshot()
    cuda = None
    try:
        import torch

        if torch.cuda.is_available():
            cuda = {"allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
                    "reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1)}
    except Exception:
        pass
    return {
        "version": VERSION,
        "installed": bypass.STATE.installed,
        "enabled": bool(getattr(shared.opts, OPTION, True)),
        "online_lora_mode": bypass.online_lora_mode(),
        "active_last_generation": _Last.active,
        "summary": _Last.summary,
        "last_generation": s,
        "cuda_memory": cuda,
    }


def _selftest(n: int = 64, tokens: int = 256, seed: int = 0, target: str = "unet"):
    from fastapi import HTTPException

    from lib_lora_bypass import selftest
    from modules.call_queue import queue_lock

    with queue_lock:
        sd = shared.sd_model
        fo = getattr(sd, "forge_objects_after_applying_lora", None) or getattr(sd, "forge_objects", None)
        if fo is None:
            raise HTTPException(status_code=409, detail="no model loaded")
        if target == "unet":
            patcher = fo.unet
        elif target == "clip":
            patcher = fo.clip.patcher if fo.clip is not None else None
        else:
            raise HTTPException(status_code=400, detail="target must be unet or clip")
        if patcher is None or not patcher.has_online_lora():
            raise HTTPException(status_code=409, detail="the current model has no on-the-fly LoRA: generate once with a "
                                                        "<lora:...> tag in a '(fp16 LoRA)' mode first")
        t0 = time.time()
        res = selftest.run(patcher, n=int(n), tokens=int(tokens), seed=int(seed), label=target)
        res["seconds"] = round(time.time() - t0, 2)
        return res


def on_app_started(_demo, app):
    if not getattr(shared.cmd_opts, "api", False):
        return
    app.add_api_route("/lora_bypass/status", _status, methods=["GET"])
    app.add_api_route("/lora_bypass/selftest", _selftest, methods=["GET"])


script_callbacks.on_app_started(on_app_started)
