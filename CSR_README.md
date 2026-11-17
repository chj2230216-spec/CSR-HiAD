# CSR-HiAD

Research extension of [cnulab/HiAD](https://github.com/cnulab/HiAD), pinned to commit `54f7c2e49956cffa231d1ec6e19a373c68a6254f`. Original Apache-2.0 license and attribution are retained when importing upstream. This is a research prototype, not a validated improvement.

## Import upstream (if not already imported)

```bash
git clone https://github.com/chj2230216-spec/CSR-HiAD.git
cd CSR-HiAD
bash scripts/import_hiad.sh
git push origin main
```

This merges upstream Git history without overwriting the CSR extension. Install dependencies following upstream README. Use Python >=3.9, CUDA, FAISS and pretrained backbone weights as required by HiAD.

## Literature and failure-mode audit

- [HLGFA math and implementation boundary](docs/literature/HLGFA_MATH_AND_IMPLEMENTATION.md)
- [PyramidFlow math and public source audit](docs/literature/PYRAMIDFLOW_MATH_AND_SOURCE.md)
- [CSR failure analysis and diagnostic extensions](docs/literature/CSR_FAILURE_AUDIT.md)

These are static audits; no new detector or measured results are introduced. Read the failure audit with the Round 1 protocol.

## Round 1 experiment protocol

See [design audit and controlled MVTec-2K bottle/capsule protocol](docs/ROUND1_EXPERIMENT_DESIGN.md) before drawing conclusions from experiments. The two missing standardization controls and full-resolution diagnostic export are required for formal mechanism verification.

## Run controlled experiments

```bash
python runs/run_csr_patchcore.py --data_root data/RealIAD-2K --category mint --image_size 2048 --gpus 0 --mode csr
python runs/run_csr_patchcore.py --data_root data/RealIAD-2K --category mint --image_size 2048 --gpus 0 --mode fused
python runs/run_csr_patchcore.py --data_root data/RealIAD-2K --category mint --image_size 2048 --gpus 0 --mode concat
```

For 4096 use `--image_size 4096` (default scales 0 1 2, 8 SCA tasks); 2048 defaults to scales 0 1, 4 SCA tasks. The thumbnail stays a standard PatchCore detector. Use `--no-use_thumbnail` to remove it; `--assignment o2o` or `a2o` to change assignment; `--calibration none` to disable normal-only task calibration. For a 384-dimensional matched comparison the scales use 192 (2K) or 128 (4K) dimensions each. Same SCA descriptors, backbone, projection and candidate bank cap across modes, although each mode constructs its own coreset.

Modes: fine (native HR), fused (HiAD weighted representation), concat (direct scale concatenation), residual (coarsest plus adjacent differences), conditional (ridge innovation), csr (conditional + normal diagonal standardization). The feature count is bounded by a random-priority uniform reservoir per task; default fit 20,000 rows, bank candidates 50,000 before upstream coreset sampling, unlike unrestricted original HiAD. For exact upstream baseline use its unmodified runner. Tail calibration splits by original image before task creation and uses only normal held-out images. It calibrates per-task patch maps before stitching and max, not image-level decision thresholds. No synthetic anomaly training/calibration.

Default pixel metrics reproduce upstream 512 resize; use `--metric_size 2048` for full 2K (memory intensive). Keep different experiment settings in distinct checkpoint roots; evaluation-only requires matching settings. GPU end-to-end training and accuracy are not yet verified. Current scope: square images, PatchCore, fixed SCA/A2O/O2O, normal-only calibration. Rectangular grids, PaDiM, cross-task caching, image-level calibration and VarAD are not implemented.

## Test core math
```bash
python -m pytest -q tests/test_csr_model.py
```
