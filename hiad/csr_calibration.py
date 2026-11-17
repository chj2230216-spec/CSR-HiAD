"""Fit task-level empirical score distribution on held-out normal images."""
import os
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from hiad.csr_model import Reservoir
from hiad.detectors.hr_csr_patchcore import HRCSRPatchCore


def calibrate_tasks(tasks, samples, config, checkpoint_root, gpu_id, batch_size,
                    logger, seed=42, rows=100000, pixels_per_map=4096):
    if not samples or any(s.label != 0 for s in samples):
        raise ValueError("calibration requires a disjoint normal image set")
    device = torch.device(f"cuda:{gpu_id}")
    for task in tasks:
        name = task["name"]
        cfg = copy.deepcopy(config.thumbnail if name == "thumbnail" else config.patch)
        if name == "thumbnail":
            cfg.patch_size = cfg.pop("thumbnail_size")
        model = HRCSRPatchCore(**cfg, logger=logger, device=device, seed=seed)
        path = os.path.join(checkpoint_root, name + "_weight.pkl")
        model.load_checkpoint(path)
        reservoir = Reservoir(rows, seed)
        rng = np.random.default_rng(seed)
        for sample in samples:
            sample.open()
            try:
                if name == "thumbnail":
                    patches = [sample.down_sampling_to_LR(task["thumbnail_size"])]
                else:
                    patches = []
                    for idx in task["indexes"]:
                        patch = sample[idx.main_index]
                        for low in idx.low_resolution_indexes or []:
                            patch.add_low_resolution_images(low, sample.image)
                        patches.append(patch)
                loader = DataLoader(model.create_dataset(patches, False, "val"),
                                    batch_size=batch_size, shuffle=False, num_workers=0)
                for mask in model.inference_step(loader, "val"):
                    flat = np.asarray(mask).ravel()
                    ix = rng.choice(len(flat), min(pixels_per_map, len(flat)), replace=False)
                    reservoir.add(flat[ix, None])
            finally:
                sample.close()
        model.csr_calibration = np.sort(reservoir.value().ravel())
        model.save_checkpoint(path)
        logger.info("Calibrated %s on %d normal images", name, len(samples))
        del model
        torch.cuda.empty_cache()
