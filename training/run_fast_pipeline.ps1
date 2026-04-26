# ~1-hour SFT+RL demo for all three scenarios (tune --base-model and counts if you miss the budget).
# Requires: pip install -r requirements.txt; .env with HF_TOKEN and HF_USERNAME for Hub upload.
# README recommends WSL/Linux for training; on Windows native, CPU-only runs may exceed 1h on 1.5B.

$ErrorActionPreference = "Stop"
# Avoid trl/transformers reading UTF-8 templates as cp1252 on Windows.
$env:PYTHONUTF8 = "1"
Set-Location (Split-Path -Parent $PSScriptRoot)

$repoRoot = (Get-Location).Path
# Local weights: set MODEL_PATH in .env or environment to the folder that contains config.json
# (e.g. ...\checkpoints\Qwen2.5-1.5B-Instruct). Otherwise we probe a common layout under checkpoints\.
$candidates = @()
if ($env:MODEL_PATH -and $env:MODEL_PATH.Trim().Length -gt 0) {
    $candidates += $env:MODEL_PATH.Trim().Trim('"').Trim("'")
}
$candidates += @(
    (Join-Path $repoRoot "checkpoints\Qwen2.5-1.5B-Instruct"),
    (Join-Path $repoRoot "checkpoints\Qwen2.5-1.5B"),
    (Join-Path $repoRoot "checkpoints\qwen")
)
$baseModel = "Qwen/Qwen2.5-1.5B-Instruct"
foreach ($p in $candidates) {
    if ($p -and (Test-Path (Join-Path $p "config.json"))) {
        $baseModel = $p
        break
    }
}
Write-Host "Base model path: $baseModel"

python -m training.export_heuristic_sft --seeds 4

# RTX 5060 (CUDA): local 1.5B + LoRA is fine; raise --batch-size to 2 if VRAM allows OOM-free runs.
# CPU: switch hub id to Qwen/Qwen2.5-0.5B-Instruct and cut --sft-max-samples / --episodes.
python -m training.train_local --run-all `
  --report-to none `
  --base-model "$baseModel" `
  --sft-max-samples 600 `
  --epochs 1 `
  --sft-warmup-steps 12 `
  --batch-size 1 `
  --grad-accum 4 `
  --max-seq-len 1024 `
  --max-new-tokens 24 `
  --episodes 18 `
  --eval-every 9 `
  --eval-max-seeds 3 `
  --sft-save-steps 50 `
  --checkpoint-every 3 `
  --grpo-update-every 6 `
  --group-size 2 `
  --lr-sft 2e-4 `
  --lr-rl 5e-6
