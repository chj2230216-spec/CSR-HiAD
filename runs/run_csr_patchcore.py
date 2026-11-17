# Modified from upstream runs/run_patchcore.py (Apache-2.0); CSR runner.
import copy
import random
import math
import json
import shutil
import yaml
import argparse
import os
from functools import partial
from easydict import EasyDict
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from hiad.utils.base import read_meta_file, create_logger, PrintTasks
from hiad.utils.split_and_gather import HRSample, MultiResolutionHRImageSpliter
from hiad.utils import metrics_gpu
from hiad.task import A2OTaskGenerator, O2OTaskGenerator, NeighborhoodTaskGenerator, SpatialClusteringTaskGenerator, RetrieveTaskGenerator
from hiad.trainer import HRTrainer
from hiad.detectors import HRPatchCore
from hiad.detectors.hr_csr_patchcore import HRCSRPatchCore
from hiad.csr_calibration import calibrate_tasks



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="High resolution Anomaly Detection")
    parser.add_argument("--data_root", default="data/MVTec-2K", help='dataset path')
    parser.add_argument("--category", default="bottle", help='category to detect')

    parser.add_argument("--image_size", default = 2048, type=int, help='Detection resolution: images will be resized to this size for detection')
    parser.add_argument("--patch_size", default = 512, type=int, help='Patch size: size of the image patches after cropping')
    parser.add_argument("--stride", default = -1, type=int, help='Stride size: set to -1 to use the same value as the patch size')
    parser.add_argument("--ds_factors", default = None, nargs='+', type=int, help='Downsampling ratio used for feature fusion')
    parser.add_argument("--fusion_weights", default = None, nargs='+', type=float, help='Fusion weights for feature fusion')
    parser.add_argument("--batch_size", default = 8, type = int, help='Batch size')

    parser.add_argument("--use_thumbnail", action=argparse.BooleanOptionalAction, default=True, help='Enable thumbnail detection (low-resolution detection)')
    parser.add_argument("--thumbnail_size", default = 512, type=int, help='low-resolution detection resolution')
    parser.add_argument("--thumbnail_config", default = 'configs/patchcore.yaml', type= str, help='Low-res(thumbnail) detection model configuration path')

    parser.add_argument("--detector_number", default = None, type=int, help='number of detector')
    parser.add_argument("--evaluation_only", action='store_true', help='Run evaluation only')
    parser.add_argument("--early_stop_epochs", default = -1, type = int,
                        help='Controls early stopping for detector: if a detector shows no performance improvement on the validation set for N epochs, training will be stopped. If -1, Early stopping will be disabled.')

    parser.add_argument("--detector_config_path", default = 'configs/patchcore.yaml', type = str, help='High-res detection model configuration path')
    parser.add_argument("--seed", default = 42, type = int)

    parser.add_argument("--checkpoint_root", default ='results/csr_checkpoints', type = str, help='Directory for saving model checkpoints')
    parser.add_argument("--log_root", default = 'results/patchcore_logs', type = str, help='Directory for saving logs')
    parser.add_argument("--vis_root", default = 'results/patchcore_vis', type = str, help='Directory for saving visualization results')

    parser.add_argument("--gpus", default="0", type=str, help='gpu ids')

    parser.add_argument("--mode", choices=["fine","fused","concat","residual","conditional","csr"], default="csr")
    parser.add_argument("--assignment", choices=["sca","o2o","a2o"], default="sca")
    parser.add_argument("--embedding_dim", type=int, default=384)
    parser.add_argument("--csr_ridge", type=float, default=0.01)
    parser.add_argument("--csr_std_floor", type=float, default=0.01)
    parser.add_argument("--csr_fit_rows", type=int, default=20000)
    parser.add_argument("--csr_bank_rows", type=int, default=50000)
    parser.add_argument("--calibration", choices=["none","tail"], default="tail")
    parser.add_argument("--calibration_ratio", type=float, default=0.2)
    parser.add_argument("--metric_size", type=int, default=512)
    args = parser.parse_args()
    if args.image_size < args.patch_size or args.patch_size < 1:
        parser.error("image size must be >= positive patch size")
    if args.stride != -1 and not 0 < args.stride <= args.patch_size:
        parser.error("invalid stride")
    if args.ds_factors is None:
        max_k = min(2, int(math.log2(args.image_size // args.patch_size)))
        args.ds_factors = list(range(max_k+1))
    if args.ds_factors != sorted(set(args.ds_factors)) or args.ds_factors[0] != 0:
        parser.error("ds_factors must be sorted unique nonnegative scales starting at zero")
    if args.patch_size * 2**args.ds_factors[-1] > args.image_size:
        parser.error("coarsest context exceeds image size")
    if args.mode not in ("fine","fused") and (len(args.ds_factors)<2 or args.embedding_dim % len(args.ds_factors)):
        parser.error("CSR modes require >=2 scales and divisible embedding_dim")
    if args.fusion_weights is None:
        args.fusion_weights = [1/len(args.ds_factors)] * len(args.ds_factors)
    if len(args.fusion_weights) != len(args.ds_factors) or min(args.fusion_weights)<0 or sum(args.fusion_weights)<=0:
        parser.error("invalid fusion weights")
    if args.detector_number is None:
        args.detector_number = 4 if args.image_size <= 2048 else 8
    if not 0 < args.calibration_ratio < 1:
        parser.error("calibration ratio must be in (0,1)")
    args.checkpoint_root = os.path.join(args.checkpoint_root, args.mode, str(args.image_size), args.assignment)
    args.log_root = os.path.join(args.log_root, args.mode, str(args.image_size), args.assignment)
    args.vis_root = os.path.join(args.vis_root, args.mode, str(args.image_size), args.assignment)
    args.gpus = [int(gpu_id) for gpu_id in args.gpus.split(',')]

    image_size = args.image_size
    patch_size = args.patch_size
    stride = args.stride

    category = args.category
    data_root = args.data_root

    detector_number = args.detector_number
    early_stop_epochs = args.early_stop_epochs

    args.checkpoint_root = os.path.join(args.checkpoint_root, args.category)
    args.log_root = os.path.join(args.log_root, args.category)
    args.vis_root = os.path.join(args.vis_root)

    os.makedirs(args.checkpoint_root, exist_ok=True)
    os.makedirs(args.log_root, exist_ok=True)
    os.makedirs(args.vis_root, exist_ok=True)

    if not args.evaluation_only:
        with open(os.path.join(args.checkpoint_root, 'args.json'), 'w+') as f:
            json.dump(vars(args), f, indent=4)

    shutil.copy(args.detector_config_path,
                os.path.join(args.checkpoint_root, os.path.basename(args.detector_config_path)))

    if args.use_thumbnail and args.thumbnail_config is not None:
        shutil.copy(args.thumbnail_config,
                    os.path.join(args.checkpoint_root, os.path.basename(args.thumbnail_config)))

    with open(args.detector_config_path) as f:
        patch_detector_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    if args.use_thumbnail:
        if args.thumbnail_config is not None:
            with open(args.thumbnail_config) as f:
                thumbnail_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        else:
            thumbnail_config = copy.deepcopy(patch_detector_config)

        thumbnail_size = args.thumbnail_size if args.thumbnail_size is not None else args.patch_size
        thumbnail_config.thumbnail_size = thumbnail_size

        config = EasyDict(patch=patch_detector_config, thumbnail=thumbnail_config)
    else:
        config = EasyDict(patch=patch_detector_config)

    config.patch.patch_size = args.patch_size
    for cfg in [config.patch] + ([config.thumbnail] if args.use_thumbnail else []):
        cfg.target_embed_dimension = args.embedding_dim
        cfg.csr_mode = args.mode if cfg is config.patch else "fused"
        cfg.csr_ridge = args.csr_ridge
        cfg.csr_std_floor = args.csr_std_floor
        cfg.csr_fit_rows = args.csr_fit_rows
        cfg.csr_bank_rows = args.csr_bank_rows

    main_logger = create_logger("main", os.path.join(args.log_root, 'main.log'), print_console=True)
    main_logger.info(args)
    main_logger.info(config)

    train_meta_path = os.path.join(data_root, category, 'train.jsonl')
    test_meta_path = os.path.join(data_root, category, 'test.jsonl')

    train_samples = read_meta_file(train_meta_path)
    test_samples = read_meta_file(test_meta_path)

    train_samples = [HRSample(image = os.path.join(args.data_root, sample['filename']),
                              image_size = image_size,
                              clsname = sample['clsname'],
                              label = sample['label'],
                              label_name = sample['label_name']) for sample in train_samples]

    test_samples = [HRSample( image = os.path.join(args.data_root,sample['filename']),
                              mask = os.path.join(args.data_root, sample['mask']) if 'mask' in sample else None,
                              image_size = image_size,
                              clsname = sample['clsname'],
                              label = sample['label'],
                              label_name = sample['label_name']) for sample in test_samples]

    main_logger.info(f"train dataset len is: {len(train_samples)}")
    main_logger.info(f"test dataset len is: {len(test_samples)}")

    train_samples = [s for s in train_samples if s.label == 0]
    if len({s.get_image_path() for s in train_samples}) != len(train_samples):
        raise ValueError("duplicate training image path")
    if set(s.get_image_path() for s in train_samples) & set(s.get_image_path() for s in test_samples):
        raise ValueError("train/test image paths overlap")
    calibration_samples = []
    if args.calibration == "tail" and not args.evaluation_only:
        if len(train_samples) < 3:
            raise ValueError("need >=3 normal images for disjoint calibration")
        random.Random(args.seed).shuffle(train_samples)
        n = min(len(train_samples)-2, max(1, round(len(train_samples)*args.calibration_ratio)))
        calibration_samples, train_samples = train_samples[:n], train_samples[n:]
    if not args.evaluation_only:
        with open(os.path.join(args.checkpoint_root, "normal_split.json"), "w") as f:
            json.dump({"fit": [s.get_image_path() for s in train_samples],
                       "calibration": [s.get_image_path() for s in calibration_samples]}, f, indent=2)
    detector_class = HRCSRPatchCore
    main_logger.info(f"Anomaly detector is: {detector_class}")

    if not args.evaluation_only:

        image_indexes = MultiResolutionHRImageSpliter(image_size = image_size, patch_size=patch_size,
                                                      ds_factors = args.ds_factors,
                                                      stride = stride if stride != -1 else None)

        if args.assignment == "o2o":
            generator = O2OTaskGenerator()
        elif args.assignment == "a2o":
            generator = A2OTaskGenerator()
        else:
            generator = SpatialClusteringTaskGenerator(HRPatchCore, config, args.batch_size,
                cluster_number=min(args.detector_number,len(image_indexes)),
                fusion_weights=args.fusion_weights, device=args.gpus)
        tasks = generator.create_tasks(train_samples, image_indexes)
        PrintTasks(tasks)
    else:
        tasks = None

    trainer = HRTrainer(batch_size = args.batch_size,
                        tasks = tasks,
                        checkpoint_root = args.checkpoint_root,
                        log_root = args.log_root,
                        vis_root = args.vis_root,
                        seed = args.seed,
                        detector_class = detector_class,
                        config = config,
                        fusion_weights = args.fusion_weights,
                        early_stop_epochs = early_stop_epochs)

    if not args.evaluation_only:
        trainer.train(
                      train_samples = train_samples,
                      gpu_ids = args.gpus,
                      main_logger=main_logger,
        )

    if not args.evaluation_only and args.calibration == 'tail':
        calibrate_tasks(trainer.tasks, calibration_samples, config, args.checkpoint_root,
                        args.gpus[0], args.batch_size, main_logger, args.seed)

    evaluators = [
        metrics_gpu.compute_imagewise_metrics_gpu,
        partial(metrics_gpu.compute_pixelwise_metrics_gpu_resize, resize = args.metric_size),
        partial(metrics_gpu.compute_pro_gpu_resize, resize = args.metric_size),
    ]

    trainer.inference(test_samples = test_samples,
                      evaluators = evaluators,
                      main_logger=main_logger,
                      gpu_ids = args.gpus,
                      vis_size = 1024)