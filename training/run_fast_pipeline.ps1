# Target: ~1 hour wall-clock for export + SFT + RL on all three scenarios (tight on slower ~25–30 s/SFT-step rigs).
# Telemetry: [SFT] % / steps left / elapsed / ETA; [RL] same + episodes + env steps; [pipeline] phase totals.
# If CUDA OOM, use --batch-size 1 and --sft-max-samples 64 in the python line below.

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
Set-Location (Split-Path -Parent $PSScriptRoot)

$repoRoot = (Get-Location).Path
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

python -m training.export_heuristic_sft --seeds 2

# SFT: few rows + short ctx + larger microbatch => very few optimizer steps.
# RL: 6 episodes = two full cycles over the three scenarios; skip mid-run eval (999); one GRPO batch at ep 6.
python -u -m training.train_local --run-all `
  --report-to none `
  --live-telemetry `
  --rl-env-log-every 5 `
  --base-model "$baseModel" `
  --sft-max-samples 96 `
  --epochs 1 `
  --sft-warmup-steps 4 `
  --sft-logging-steps 2 `
  --sft-save-steps 20 `
  --batch-size 2 `
  --grad-accum 4 `
  --max-seq-len 512 `
  --max-new-tokens 20 `
  --episodes 6 `
  --eval-every 999 `
  --eval-max-seeds 1 `
  --checkpoint-every 2 `
  --grpo-update-every 6 `
  --group-size 2 `
  --max-prompts-per-update 64 `
  --lr-sft 2e-4 `
  --lr-rl 5e-6
