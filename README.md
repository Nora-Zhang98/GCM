# Human-Inspired Scene Understanding: A Grounded Cognition Method for Unbiased Scene Graph Generation (TPAMI25)

This repository contains the official code implementation for the [paper](https://ieeexplore.ieee.org/document/11264347)

## Installation
Check [INSTALL.md](./INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](./DATASET.md) for instructions of dataset preprocessing.

## Pretrained Model
You can download the pretrained Faster R-CNN from the following links: [VG150](https://1drv.ms/u/s!AmRLLNf6bzcir8xemVHbqPBrvjjtQg?e=hAhYCw),
[GQA200](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21779870&authkey=AH5CPVb9g5E67iQ),
[OIV6](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EfGXxc9byEtEnYFwd0xdlYEBcUuFXBjYxNUXVGkgc-jkfQ?e=lSlqnz)

## Train
We provide [scripts](./scripts/train.sh) for training the models (multiGPU).

Here is the example of VTransE-GCM for PredCls on VG150 dataset. 
### for VG150
```
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port 10025 --nproc_per_node=2 \
  tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR VTransEPredictorGCM \
  MODEL.TRAIN_INFER False \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 2 \
  SOLVER.MAX_ITER 45000 SOLVER.BASE_LR 5e-4 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS "(10000, 16000)" SOLVER.VAL_PERIOD 5000 \
  SOLVER.CHECKPOINT_PERIOD 5000 GLOVE_DIR tools \
  MODEL.PRETRAINED_DETECTOR_CKPT ../pretrained_ckpt/model_final.pth \
  OUTPUT_DIR tools/output/relation_baseline \
  GLOBAL_SETTING.DATASET_CHOICE VG \
  SOLVER.PRE_VAL False \
  SOLVER.GRAD_NORM_CLIP 5.0;
```
### for GQA200
```
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port 10025 --nproc_per_node=2 \
  tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR VTransEPredictorGCM \
  MODEL.TRAIN_INFER False \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 2 \
  SOLVER.MAX_ITER 60000 SOLVER.BASE_LR 1e-4 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS "(10000, 16000)" SOLVER.VAL_PERIOD 5000 \
  SOLVER.CHECKPOINT_PERIOD 5000 GLOVE_DIR tools \
  MODEL.PRETRAINED_DETECTOR_CKPT_GQA ../pretrained_ckpt/gqa_model_final_from_vg.pth \
  OUTPUT_DIR tools/output/relation_baseline \
  GLOBAL_SETTING.DATASET_CHOICE GQA_200 \
  SOLVER.PRE_VAL False \
  SOLVER.GRAD_NORM_CLIP 5.0;
```

### for OIV6
```
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port 10025 --nproc_per_node=2 \
  tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR VTransEPredictorGCM \
  MODEL.TRAIN_INFER False \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 2 \
  SOLVER.MAX_ITER 60000 SOLVER.BASE_LR 1e-3 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS "(10000, 16000)" SOLVER.VAL_PERIOD 5000 \
  SOLVER.CHECKPOINT_PERIOD 5000 GLOVE_DIR tools \
  MODEL.PRETRAINED_DETECTOR_CKPT_OIV6 ../pretrained_ckpt/oiv6_det.pth \
  OUTPUT_DIR tools/output/relation_baseline \
  GLOBAL_SETTING.DATASET_CHOICE OIV6 \
  GLOBAL_SETTING.CHOOSE_BEST_MODEL_BY_METRIC _recall
  SOLVER.PRE_VAL False \
  SOLVER.GRAD_NORM_CLIP 5.0;
```

We also provide mean and var for VTransE, MOTIFS, VCTree, and PENet among VG, GQA200 and OIV6 in this [link](https://1drv.ms/f/c/60174365786eb250/IgCVoNRaCGy_TqiMy5QEZVP2AZNnuWsW1Kq1_PyYP3gqAp8?e=ZVlgoK), if you want to obtain them for other methods, train the baseline method and then run [relation_infer_train.py](./tools/relation_infer_train.py)
to restore rel_reps. (path: cfg.OUTPUT_DIR/infer_train_feat)
```
python3 \
  tools/relation_infer_train.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetworkGCM \
  DTYPE "float32" \
  TEST.IMS_PER_BATCH 4 \
  GLOVE_DIR . \
  MODEL.PRETRAINED_DETECTOR_CKPT ../../pretrained_ckpt/model_final.pth \
  OUTPUT_DIR . \
  GLOBAL_SETTING.DATASET_CHOICE VG \
  TRAIN_INFER True \
  SOLVER.GRAD_NORM_CLIP 5.0;
```
After that, run [cal_mean_std.py](./tools/cal_mean_std.py), then mean & var is calculated.

## Acknowledgement

The code is implemented based on [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) and [SHA-GCL](https://github.com/dongxingning/SHA-GCL-for-SGG/tree/master).


## Citation
```
@ARTICLE{11264347,
  author={Zhang, Ruonan and Hao, Yiqing and Zhang, Feng and An, Gaoyun and Song, Binyang and Wu, Dapeng Oliver},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Human-Inspired Scene Understanding: A Grounded Cognition Method for Unbiased Scene Graph Generation}, 
  year={2025},
  volume={},
  number={},
  pages={1-18},
  keywords={Cognition;Linguistics;Visualization;Semantics;Heavily-tailed distribution;Telecommunication traffic;Communication switching;Visual perception;Context modeling;Brain modeling;Scene Graph Generation;Scene Understanding;Grounded Cognition;Multimodal Perception;Shapley Value},
  doi={10.1109/TPAMI.2025.3635152}}
```
