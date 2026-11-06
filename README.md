# LoRA Bypass for Forge Neo: full-speed, exact LoRA for int8 / fp8 / GGUF models that don't fit in VRAM (12–16 GB GPUs)

> **In short.** An extension for Stable Diffusion WebUI Forge Neo 2.30. In Diffusion in Low Bits = "… (fp16 LoRA)" it computes every LoRA as a separate low-rank branch, `y = W·x + scale·B(A·x)`, instead of rebuilding the full weight on every step. LoRA generations run almost as fast as generations without LoRA (Krea 2 on a 12 GB card: 40.1 s → 25.4 s), stay exact and give byte-identical images for the same seed. **Required launch flag for the target setup: `--pin-shared-memory`** ([why](#required-launch-flag---pin-shared-memory)).

## The problem

When the diffusion model does not fit into VRAM (for example Krea 2 int8, 12.8 GB, on a 12 GB RTX 4070), Forge Neo offers two ways to apply a LoRA, and both hurt:

- **Automatic (merged)** bakes the LoRA into the weights. With a partially loaded model the VRAM / RAM split shifts by a few megabytes between runs, so the same seed gives a different image every time, and the LoRA is rounded into the quantized weights.
- **Automatic (fp16 LoRA)** (on-the-fly) is exact and stable, but slow: on every step, for every LoRA layer, Forge dequantizes the whole base weight, adds the LoRA delta and multiplies on the slow bf16 path instead of the fast int8 / fp8 kernel. A generation takes about 1.7× as long as without LoRA.

## How it works

- The base layer keeps running on Forge's own fast int8 / fp8 / GGUF kernel; the LoRA is added as a thin side branch at full precision: `y = W·x + scale·B(A·x)`.
- It hooks in through Forge's own object patches on `Linear.forward`. LoRA loading, `<lora:…>` tags and weights work as usual; switching the bypass on or off needs no LoRA reload.
- Anything that cannot be bypassed automatically stays on the stock on-the-fly path, so the result is always correct (see [below](#what-stays-on-stock-on-the-fly)).
- In Automatic (merged) mode the extension does nothing.

## Results

RTX 4070 12 GB, 32 GB RAM, Windows 11; warm runs, time per image.

| Model + LoRA | Settings | Stock on-the-fly | **With bypass** |
|---|---|---|---|
| Krea 2 int8 (12.8 GB) + LoRA 111 MB | 1152×1536, ER SDE, 8 steps | 40.1 s | **25.4 s** |
| Krea 2 int8 + LoRA 914 MB | same | 41.1 s | **26.1 s** |
| Krea 2 int8 + LoRAs 914 MB + 111 MB | same | 43.6 s | **28.4 s** |
| Qwen-Image 2.1 int8 + LoRA | 1024×1024, 16 steps | 35.9 s | **22.9 s** |
| Z-Image Turbo int8 + LoRA | 1024×1536, 9 steps | 19.2 s | **12.8 s** |
| MOSS-Image int8 + LoRA | 1024×1024, 24 steps | 69.3 s | **47.9 s** |
| SDXL fp16 + LoRA (nothing to gain) | 832×1216, 24 steps | 10.7 s | 10.7 s |

For reference, Krea 2 without LoRA takes 23.4 s and Automatic (merged) 28.9 s.

- **Stability:** the same seed gives a byte-identical image across runs, also after moving the VRAM / RAM boundary (`--reserve-vram 2`). In merged mode the same generation changed from run to run.
- **Quality:** the same frame as stock on-the-fly (pose, face, clothes, background). The LoRA term is computed 2–3× more precisely than on-the-fly, which rounds `W + ΔW` to bf16.
- Full test report with comparison grids: [TEST_REPORT.md](TEST_REPORT.md).

## Who it is for

- GPUs where the diffusion model is larger than roughly VRAM − 4 GB: models from ~8 GB on 12 GB cards, from ~12 GB on 16 GB cards, from ~20 GB on 24 GB cards, used with Diffusion in Low Bits = "Automatic (fp16 LoRA)" (or another "… (fp16 LoRA)" mode). This is exactly where merged mode is non-deterministic and stock on-the-fly is slow.
- Any int8 / fp8 / GGUF model with LoRA: the LoRA is computed at full precision instead of being rounded into the quantized weight.
- No benefit for fp16 / bf16 models: there is nothing to dequantize, and stock on-the-fly is already cheap.

## Required launch flag: `--pin-shared-memory`

In the target setup the model does not fit into VRAM, so Forge keeps part of it in RAM during generation and moves the whole model to RAM after each image to make room for the VAE. With Krea 2 and its text encoder the Forge process holds about 33 GB of memory. If you use other programs between generations (browser, video), Windows pages the model weights out to compressed memory and the page file, and the next generation first waits for them to come back. This is Forge and Windows behaviour, not the extension's: in tests the bypass stayed active on all layers, and stock on-the-fly slowed down even more.

Forge's own flag `--pin-shared-memory` pins the offloaded weights in RAM, so Windows cannot page them out. Add it to the `COMMANDLINE_ARGS` line of `webui-user.bat` and restart Forge:

```bat
set COMMANDLINE_ARGS=<your arguments> --pin-shared-memory
```

On startup the Forge console shows `Pinned Memory: … MB`. Forge caps pinned memory at 45% of RAM on Windows (14.7 GB with 32 GB RAM). Images are byte-identical with and without the flag, and speed under normal use is the same (±1 s).

Krea 2, LoRA 111 MB, 1152×1536, 8 steps:

| Conditions | Bypass | Bypass + flag | Stock on-the-fly | On-the-fly + flag |
|---|---|---|---|---|
| Generations back to back | 26.2 s | 26.6 s | 43.7 s | 41.5 s |
| Another program took RAM, then closed | 31.8 s | **26.7 s** | 52.5 s | 46.0 s |
| Another program holds RAM during generation | 34.5 s | 30.1 s | 43.4 s | 43.0 s |
| Another program holds 1.5 GB of VRAM | 31.3 s | 29.5 s | 66.3 s | 50.2 s |

Limitations: the flag does not pin the text encoder, so the first generation with a new prompt after a long time in other programs may still take a few seconds longer (up to ~15 s to reload the encoder after heavy memory pressure). Other programs cannot use the pinned memory, so when RAM runs short their data goes to the page file instead. With 32 GB RAM memory is tight for models of Krea 2 size; the radical fix is more RAM or closing heavy programs before generating.

## Install (Windows)

Close Forge, then run in PowerShell from the repository folder:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\verify.ps1
```

Forge is expected in `C:\AI\Forge Neo`; otherwise add `-ForgeDir "path"`. The script copies `sd-forge-lora-bypass` into `extensions\` and writes the manifest `.lora-bypass-install.json`; Forge's code, git branches, `config.json` and venv are not touched. The installer refuses to run if Forge is running, if another version is already installed (run `rollback.ps1` first) or if the extension folder was not created by `install.ps1`.

Manual install: copy the `sd-forge-lora-bypass` folder into Forge's `extensions` folder (tested on Windows only).

Then add `--pin-shared-memory` to the launch arguments (see [above](#required-launch-flag---pin-shared-memory)).

## Usage

- Set Diffusion in Low Bits to "Automatic (fp16 LoRA)" (or another "… (fp16 LoRA)" mode).
- Toggle: Settings → LoRA Bypass → "Enable LoRA bypass (fp16 LoRA modes)", on by default. It applies from the next generation; no LoRA reload is needed.
- To put the toggle in the top bar: Settings → User Interface → Quicksettings list → add `lora_bypass_enabled`, Apply settings, Reload UI. The checkbox appears under the Checkpoint / Diffusion in Low Bits row.
- When active, the console prints a line like `[LoRA Bypass] UNet: 256/256 LoRA layers bypassed` after LoRAs are loaded, and the image parameters (PNG Info) get `LoRA bypass: on`. With the toggle off or in merged mode the field is absent.
- Per-generation statistics: Settings → LoRA Bypass → "Print per-generation bypass statistics to the console".
- Numeric self-check (only when Forge runs with `--api`): after a generation with LoRA open `http://127.0.0.1:7860/lora_bypass/selftest`. The extension compares up to 64 layers with a float32 reference and returns `"passed": true/false`. `GET /lora_bypass/status` shows the current state.

## Uninstall

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\rollback.ps1
```

Removes exactly what `install.ps1` installed (according to the manifest) and the manifest itself. Manually: close Forge and delete the `extensions\sd-forge-lora-bypass` folder and the `.lora-bypass-install.json` file in the Forge folder. The `lora_bypass_enabled` / `lora_bypass_verbose` keys left in `config.json` have no effect without the extension.

## What stays on stock on-the-fly

Bypass is used for linear layers where all LoRAs are plain LoRAs. Everything else is computed the stock way: the result is correct, just without the speed-up for those layers.

- LoKr, LoHa, DoRA, GLoRA, OFT, full `diff` patches, LoRA on bias, convolution layers (LoCon in SDXL).
- Layers that the model runs through a fused kernel that bypasses `forward`: `ffn_down` in MOSS-Image with Fused SiTU (MOSS-Image Speed Boost) and similar `linear_input_act` kernels.
- The built-in LoRA Control (`<lora:name:[1@0,0@1]>`) works with bypass too: the strength is read on every step.

## Compatibility

Tested on Forge Neo 2.30 (Windows 11, Python 3.13, torch 2.13 + cu130) with Krea 2, Qwen-Image 2.1, Z-Image, MOSS-Image (with and without Speed Boost) and SDXL (Pony / Illustrious), together with ADetailer, Batch count, LoRA Control and two LoRAs at once. After a Forge update run `verify.ps1`: it checks that the Forge code the extension hooks into is still there.

## License

AGPL-3.0, same as Forge (`LICENSE`).
