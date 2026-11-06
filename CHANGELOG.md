# Changelog

## 1.0.0 — 2026-11-06

First release.

- Extension `sd-forge-lora-bypass` for Forge Neo 2.30: in Diffusion in Low Bits "… (fp16 LoRA)" modes (and for GGUF) LoRA on linear layers is computed as a separate low-rank branch `y = W·x + scale·B(A·x)`; the base layer stays on the fast int8 / fp8 / fp16 path.
- LoRA registration in Forge is unchanged: only `forward` of eligible linear layers is replaced, through Forge's own object patches. Everything else (LoKr, LoHa, DoRA, convolutions, LoRA on bias, fused `linear_input_act` kernels) runs on the stock on-the-fly path.
- Toggle in Settings → LoRA Bypass (on by default); it applies from the next generation without reloading LoRAs and can be added to Quicksettings.
- A console line after LoRAs are loaded and a `LoRA bypass: on` field in the infotext.
- API (only with `--api`): `GET /lora_bypass/status`, `GET /lora_bypass/selftest` — numeric layer check against a float32 reference.
- `install.ps1` / `verify.ps1` / `rollback.ps1` scripts with the `.lora-bypass-install.json` manifest.
- Docs: Forge's `--pin-shared-memory` flag is required when the model does not fit into VRAM; measurements of the slowdown after using other programs and of the fix (README, TEST_REPORT).
