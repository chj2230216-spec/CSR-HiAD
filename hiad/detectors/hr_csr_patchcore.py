"""CSR-PatchCore extension of cnulab/HiAD (Apache-2.0).
Modified PatchCore embedding routine below preserves upstream patchification.
Upstream: https://github.com/cnulab/HiAD/tree/54f7c2e49956cffa231d1ec6e19a373c68a6254f
"""
import numpy as np
import torch
import torch.nn.functional as F
from .hr_patchcore import HRPatchCore
from hiad.utils.split_and_gather import HRImageIndex
from hiad.csr_model import CSRTransform, Reservoir, tail_score


class HRCSRPatchCore(HRPatchCore):
    def __init__(self, *args, csr_mode="csr", csr_ridge=0.01,
                 csr_std_floor=0.01, csr_fit_rows=20000, csr_bank_rows=50000,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if csr_mode not in ("fine", "fused", *CSRTransform.MODES):
            raise ValueError("unknown CSR mode")
        self.csr_mode = csr_mode
        self.csr_ridge, self.csr_std_floor = csr_ridge, csr_std_floor
        self.csr_fit_rows, self.csr_bank_rows = csr_fit_rows, csr_bank_rows
        self.csr_transform = None
        self.csr_calibration = None
        self.csr_seed = int(kwargs.get("seed", 0))

    @torch.no_grad()
    def get_multi_resolution_aligned_embeddings(self, data):
        """Return list of scales, each a list of BCHW backbone tensors.
        Preserve upstream crop/index convention, including batch crop resizing.
        """
        base = self.embedding(data["image"].to(self.device))
        scales = [base]
        keys = sorted((k for k in data if k.startswith("low_resolution_image_")),
                      key=lambda k: int(k.rsplit("_", 1)[1]))
        for key in keys:
            outputs = self.embedding(data[key].to(self.device))
            indexes = data[key.replace("image", "index")]
            aligned = []
            for feat_batch, ref in zip(outputs, base):
                crops = []
                for feat, idx in zip(feat_batch, indexes):
                    p = HRImageIndex.from_str(idx)
                    sh = self.patch_size[1] / feat.shape[-2]
                    sw = self.patch_size[0] / feat.shape[-1]
                    x0, y0 = int(p.x/sw), int(p.y/sh)
                    x1, y1 = int(p.x/sw+p.width/sw), int(p.y/sh+p.height/sh)
                    crop = feat[:, y0:y1, x0:x1]
                    if min(crop.shape[-2:]) < 1:
                        raise ValueError("empty aligned feature crop; reduce scales")
                    crops.append(crop)
                common_shape = crops[0].shape[-2:]
                crops = [v if v.shape[-2:] == common_shape else
                         F.interpolate(v[None], size=common_shape, mode="bilinear",
                                       align_corners=False)[0] for v in crops]
                aligned.append(F.interpolate(torch.stack(crops), size=ref.shape[-2:],
                                             mode="bilinear", align_corners=False))
            scales.append(aligned)
        return scales

    def _embed_scale(self, features, detach=True, provide_patch_shapes=False):

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]

        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features

        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)
        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)



    @torch.no_grad()
    def _paired_rows(self, data):
        scales = self.get_multi_resolution_aligned_embeddings(data)
        s = len(scales)
        if s < 2 or self.target_embed_dimension % s:
            raise ValueError("CSR requires >=2 scales and embedding dim divisible by scale count")
        d = self.target_embed_dimension // s
        rows = []
        shapes = None
        for features in scales:
            z, ps = self._embed_scale(features, detach=False, provide_patch_shapes=True)
            if shapes is not None and ps != shapes:
                raise ValueError("scale spatial shape mismatch")
            shapes = ps
            z = F.adaptive_avg_pool1d(z[:, None, :], d).squeeze(1)
            rows.append(z.detach().cpu().numpy())
        return np.stack(rows, axis=1), shapes

    @torch.no_grad()
    def patchcore_emb(self, samples, detach=True, provide_patch_shapes=False):
        if self.csr_mode == "fused":
            return super().patchcore_emb(samples, detach, provide_patch_shapes)
        if self.csr_mode == "fine":
            return self._embed_scale(self.embedding(samples["image"].to(self.device)),
                                     detach, provide_patch_shapes)
        x, shapes = self._paired_rows(samples)
        if self.csr_transform is None:
            raise RuntimeError("fit or load CSR transform before inference")
        z = self.csr_transform.transform(x)
        if not detach:
            z = torch.from_numpy(z).to(self.device)
        return (z, shapes) if provide_patch_shapes else z

    def train_step(self, train_dataloader, task_name, checkpoint_path,
                   val_dataloader=None, evaluators=None):
        self.forward_modules.eval()
        # Upstream defaults to drop_last=True. Keep all normal images in each task.
        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(train_dataloader.dataset, batch_size=train_dataloader.batch_size,
                                      shuffle=False, num_workers=0, drop_last=False)
        if self.csr_mode in CSRTransform.MODES:
            fit = Reservoir(self.csr_fit_rows, self.csr_seed)
            for samples in train_dataloader:
                if "label" in samples and torch.any(samples["label"] != 0):
                    raise ValueError("CSR fitting requires normal data only")
                rows, _ = self._paired_rows(samples)
                fit.add(rows)
            self.csr_transform = CSRTransform(self.csr_mode, self.csr_ridge,
                                              self.csr_std_floor).fit(fit.value())
            del fit
        bank = Reservoir(self.csr_bank_rows, self.csr_seed + 1)
        for samples in train_dataloader:
            if "label" in samples and torch.any(samples["label"] != 0):
                raise ValueError("memory bank requires normal data only")
            bank.add(self.patchcore_emb(samples))
        candidates = bank.value()
        if int(len(candidates) * self.percentage) < self.anomaly_scorer_num_nn:
            raise ValueError("coreset too small for requested nearest-neighbor count")
        self.patch_features = self.featuresampler.run(candidates)
        self.anomaly_scorer.fit(detection_features=[self.patch_features])
        # Explicitly use only normal calibration, not upstream synthetic validation.
        if val_dataloader is not None:
            raise ValueError("use separate normal-only calibration stage")
        self.save_checkpoint(checkpoint_path)
        return True

    def patch_post_processing(self, patch_anomaly_score, eps=1e-4):
        if self.csr_calibration is not None:
            return tail_score(patch_anomaly_score, self.csr_calibration)
        return super().patch_post_processing(patch_anomaly_score, eps)

    def save_checkpoint(self, checkpoint_path):
        state = dict(patch_features=self.patch_features,
                     max_anomaly_score=self.max_anomaly_score,
                     min_anomaly_score=self.min_anomaly_score,
                     fusion_weights=self.fusion_weights,
                     csr_mode=self.csr_mode,
                     csr_transform=(None if self.csr_transform is None else
                                    self.csr_transform.state_dict()),
                     csr_calibration=self.csr_calibration,
                     csr_dimension=self.target_embed_dimension)
        torch.save(state, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        # Trusted local research checkpoints only (contains numpy arrays).
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(checkpoint_path, map_location="cpu")
        if state["csr_mode"] != self.csr_mode or state["csr_dimension"] != self.target_embed_dimension:
            raise ValueError("checkpoint mode or dimension differs from run configuration")
        self.patch_features = state["patch_features"]
        self.anomaly_scorer.fit(detection_features=[self.patch_features])
        self.fusion_weights = state["fusion_weights"]
        self.min_anomaly_score = state["min_anomaly_score"]
        self.max_anomaly_score = state["max_anomaly_score"]
        self.csr_calibration = state["csr_calibration"]
        self.csr_transform = (None if state["csr_transform"] is None else
                              CSRTransform.from_state_dict(state["csr_transform"]))
