# Shared paired-feature diagnostic: Round 1 (D0)

This is the **mechanism-screening** protocol following the literature failure audit, not a replacement for the original HiAD reproduction or the full A0–A7 Coreset comparison. No GPU benchmark has been run in the authoring environment. The default normal fit/holdout split is 80/20 by original image; all 17 scores share it. Do not compare its numbers directly against Phase A trained on all normals.

## Run

From the repository root, install the original HiAD dependencies and pretrained backbone weights. Python >=3.9, CUDA, SciPy, scikit-learn and FAISS are needed. The actual input files must exist under data/MVTec-2K/bottle and capsule with train.jsonl/test.jsonl. The runner checks labels, native 2048×2048, masks and exact file hash overlap. It does not check near-duplicates or source-image groups; check these independently before scientific use.

```bash
python -m pytest -q tests/test_csr_model.py tests/test_csr_diagnostics.py tests/test_csr_metrics.py
python runs/run_csr_diagnostics.py --category bottle --out results/diagnostic_smoke/bottle --batch_size 2 --gpu 0 --audit_alignment
GPU=0 BATCH_SIZE=2 bash scripts/run_round1_diagnostics.sh
```

The first command with bottle is a full one-category experiment, intended as first end-to-end check; it is not a tiny synthetic smoke. If it fails, inspect status.json, fix cause, use a new output directory. Batch wrapper refuses overwrites; first batch defaults seed42. For confirmation use SEEDS="2025 2026" OUT_ROOT=results/round1_shared_confirm (or SEEDS="42 2025 2026" in a new root). Optional --faiss_gpu needs GPU FAISS; CPU FAISS default is slower but valid. --save_full can consume hundreds of MB per test image; default exports exact feature-grid maps plus patch coordinates and fixed postprocessing to reconstruct full maps. --no_smoothing is a separate, all-modes sensitivity run.

## Mathematical controls

Per image, each patch's two aligned backbone outputs are extracted once (plus one upstream fused verification batch if --audit_alignment). Reuse the original PatchCore aggregation; native fine and native fused have D=384; paired fine/coarse have D/2=192. Shared fixed-row reservoir has 20,000 fit rows and 5,000 bank rows, identical IDs across all representations. Holdout rows are sampled separately; no held-out or test row fits the transform/bank. No synthetic anomalies, no thumbnail, no SCA, no tail calibration. This is not a statement that A2O is optimal.

| Score mode | Function |
|---|---|
| fine/fused | native fine or HiAD-style map fusion then aggregate, fixed uniform bank |
| paired_fine/paired_fused | same 192-d projection, fine/average |
| concat/residual/conditional/csr | original four representations, common bank |
| concat_std/residual_std | same diagonal standardization without learned prediction |
| direct_csr | direct fine prediction with B regularized toward 0, instead of B toward I |
| raw_residual_score | mean squared standardized f-c |
| innovation_score | mean squared standardized f-cB-b from residual ridge |
| direct_ridge_score | direct-ridge innovation statistical score |
| coarse_bank/innovation_bank | kNN on separate standardized CSR blocks |
| separate_sum | sum of the above two distances; scores share squared standardized units |

All neighbor distances are mean squared L2 (k=5); no per-test normalization or calibrated fusion weights. For fixed shared rows, CSR joint distance >= separate_sum for every query, up to numerical tolerance; this is a geometry check, not a performance guarantee. Separate retrieval costs two searches; total feature storage is the same as the joint bank when measured just in feature values. Combined diagnostic stores multiple banks simultaneously, so do not interpret its runtime as deployment latency.

## Output

manifest.json (including hashes and split); backbone.json (actual weights hash and coordinates); shared_rows.npz and row_ranges.json (row identities); diagnostic_state.pt (trusted local checkpoint: each transformed bank and transforms; not an original HRTrainer checkpoint); normal_diagnostics.json (independent normal R², MSE, standardization floor usage, B norms); predictions/NNNN/grid_scores.npz (16 patches, grid score for each mode); optional native .npy and mask; image_scores.csv; metrics.json / metrics.csv and each histogram .npz; runtime.json; status.json. The summary tool outputs per-category metrics, paired differences in percentage points, separate category averages, macro average only if both categories are available, and status.

Native 2048 masks are used without resize. Images are scored by the maximum valid 16×16 mean, equivalent to HiAD image pooling. All modes use same bilinear interpolation, patch Gaussian sigma4, and whole-image Gaussian sigma4 (unless --no_smoothing). Per-image streaming avoids all-test GPU tensors. P-AUC, P-AP, AUPRO@FPR0.30, small-instance PRO at FPR0.1% and 1% are **approximate**, fixed 65,536 equally spaced bins in log1p(score) in [0,30], no test-dependent scaling. Ties introduced by binning can affect metrics. Image-level AUC/AP use exact sklearn ranking. PRO uses 8-connected components; small <=419 pixels; report small_instances and overflow counts. The fixed FPR diagnostic is not a deployment threshold. Check doubled bins (--bins 131072) / exact AP on a subset before final conclusions. No exact full-dataset AP, near-duplicate audit, complete size-bin report, boundary diagnostic, normal score calibration, original Coreset, SCA or isolated profiling yet.

## Interpretation

1. Compare innovation_score vs raw_residual_score and direct_ridge_score vs innovation_score; inspect normal R², floor rates and B-I.
2. Compare csr vs concat_std/residual_std. If no advantage, condition prediction is not established.
3. Compare separate_sum vs csr and individual block scores; separate-sum improvement supports score separation as a candidate, not a guaranteed architecture.
4. Compare paired_fine/fused and native fine/fused; dimensional confounding is not removed by only comparing D=384.
5. Only after D0, proceed to the Phase A matched Coreset protocol and then fixed SCA/global branch if supported. No score is a calibrated probability. R7 local conditional distribution is intentionally not implemented before evidence of heteroscedasticity.

This diagnostic intentionally does not choose a winner using the test set or present measured results before running the actual data.
