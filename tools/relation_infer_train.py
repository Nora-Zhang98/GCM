# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')

# for baseline methods to calculate mean and var

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="../configs/e2e_relation_X_101_32_8_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    if cfg.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
        pretrain_object_detector_dir = cfg.MODEL.PRETRAINED_DETECTOR_CKPT
    elif cfg.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
        pretrain_object_detector_dir = cfg.MODEL.PRETRAINED_DETECTOR_CKPT_GQA
    elif cfg.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
        pretrain_object_detector_dir = cfg.MODEL.PRETRAINED_DETECTOR_CKPT_OIV6

    load_mapping = {"roi_heads.relation.box_feature_extractor": "roi_heads.box.feature_extractor",
                    "roi_heads.relation.union_feature_extractor.feature_extractor": "roi_heads.box.feature_extractor"}
    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)

    if checkpointer.has_checkpoint():
        _ = checkpointer.load(pretrain_object_detector_dir, update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
    else:
        # load_mapping is only used when we init current model from detection model.
        checkpointer.load(pretrain_object_detector_dir, with_optim=False, load_mapping=load_mapping)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )

    if cfg.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
        dataset_names = cfg.DATASETS.TEST
    elif cfg.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
        dataset_names = cfg.DATASETS.GQA_200_TEST
    elif cfg.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
        dataset_names = cfg.DATASETS.OIV6_TEST
    else:
        dataset_names = None
        exit('wrong Dataset name!')

    output_folders = [None] * len(dataset_names)

    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, mode="train", is_distributed=distributed)

    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


if __name__ == "__main__":
    main()
