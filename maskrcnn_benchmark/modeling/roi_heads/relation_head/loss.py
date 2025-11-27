# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy.random as npr

from maskrcnn_benchmark.layers import smooth_l1_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat

def weight_calculate(beta, REL_SAMPLES, cfg):
    effective_num = 1.0 - np.power(beta, REL_SAMPLES)
    weights = (1-beta)/np.array(effective_num)
    weights = weights / np.sum(weights) * len(REL_SAMPLES)
    file_path = '/all_mean_var_'
    if cfg.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
        file_path += 'vg_'
    elif cfg.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
        file_path += 'gqa_'
    elif cfg.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
        file_path += 'oiv6_'

    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == 'PrototypeEmbeddingNetworkGCM':
        file_path += 'penet'
    elif cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == 'MotifPredictorGCM':
        file_path += 'motifs'
    elif cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == 'VCTreePredictorGCM':
        file_path += 'vctree'
    elif cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == 'VTransEPredictorGCM':
        file_path += 'vtranse'

    class_var =  torch.load(cfg.OUTPUT_DIR + file_path + '.pkl', map_location=torch.device('cpu'))['all_var']
    class_var = F.normalize(class_var, p=2, dim=0) / 100
    weights[1:] = weights[1:] + (class_var.sum(dim=1) / 1000).numpy()
    return weights

# VG sample numbers
rel_num = torch.tensor([6144612,    5799,     226,     172,     435,     570,    1378, 1178,   11074,     553,     449,
                        1024,     396,     421, 426,       5,     897,     174,     150,     638,   49772, 7293,
                        16957,    3187,     469,     764,     207,      88, 212,   16855,   25549,   85620,     264,
                        1070,     121, 516,     349,      94,    2451,      27,    3428,    1752, 283,    3774,
                        333,     213,     991,     774,   32776, 3454,    9286])

rel_num_sggen = torch.tensor([10306722,   7274,    228,    223,    355,    756,   1786,   1333, 14373,    695,    458,
                              1349,    313,    388,    523,     10, 1029,    154,     97,    588,  65246,  10209,
                              21951,   4004, 652,   1151,    293,     69,    238,  21426,  31832, 102948, 367,   1171,
                              150,    446,    392,    134,   3079,     35, 4288,   2506,    341,   5157,    443,
                              310,   1388,   1240, 55660,   5838,  11732])

# GQA sample numbers
rel_num_gqa = torch.tensor([200000, 64218, 47205, 32126, 25203, 21104, 15890, 15676, 7688, 6966, 6596, 6044, 5250, 4260, 4180, 4131, 2859,
         2559, 2368, 2351, 2134, 1673, 1532, 1373, 1273, 1175, 1139, 1123, 1077, 941, 916, 849, 835, 808, 782, 767, 628,
         603, 569, 540, 494, 416, 412, 412, 398, 395, 394, 390, 345, 327, 302, 301, 292, 275, 270, 267, 267, 264, 258,
         251, 233, 233, 229, 224, 215, 214, 209, 204, 198, 195, 192, 191, 185, 181, 176, 158, 158, 154, 151, 148, 143,
         136, 131, 130, 130, 128, 127, 125, 124, 124, 121, 118, 112, 112, 106, 105, 104, 103, 102, 52, 52])

# OIV6 sample numbers
rel_num_oiv6 = torch.tensor([3002936, 115251, 33018, 102653, 240, 1332, 189, 67, 34684, 12223, 3460, 287, 96, 10, 3916,
                            82, 11, 20149, 87, 1797, 11, 4192, 1988, 151, 54, 950, 22, 524, 75, 10881, 160])

class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        cfg,
        attri_on,
        num_attri_cat,
        max_num_attri,
        attribute_sampling,
        attribute_bgfg_ratio,
        use_label_smoothing,
        predicate_proportion,
        use_reweight
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.cfg = cfg
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_label_smoothing = use_label_smoothing
        self.pred_weight = (1.0 / torch.FloatTensor([0.5,] + predicate_proportion)).cuda()
        self.use_reweight = use_reweight
        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:
            self.criterion_loss = nn.CrossEntropyLoss()
            if self.use_reweight == 'PENET':
                self.criterion_loss2 = VG_Loss() # VG_loss--reweight
            elif self.use_reweight == 'HTCL':
                if self.cfg.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
                    if not self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                        rels_number = rel_num_sggen
                    else:
                        rels_number = rel_num
                elif self.cfg.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
                    rels_number = rel_num_gqa
                elif self.cfg.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
                    rels_number = rel_num_oiv6
                sample_num = rels_number.clone()
                sample_num[sample_num < self.cfg.MODEL.CUT_RELS] = self.cfg.MODEL.CUT_RELS
                self.cls_weight = torch.FloatTensor(weight_calculate(self.cfg.MODEL.NUM_BETA, sample_num, cfg)).cuda()

    def __call__(self, proposals, rel_labels, relation_logits, refine_logits):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        if self.attri_on:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attri_on = False
                refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits

        # relation_logits = cat(relation_logits, dim=0) # when running baseline
        refine_obj_logits = cat(refine_obj_logits, dim=0)
        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)

        if self.use_reweight == 'PENET':
            loss_relation = self.criterion_loss2.pred_loss(rel_labels.long(), relation_logits, alpha=2)  # VG_loss--reweight
        elif self.use_reweight == 'HTCL':
            loss_relation = F.cross_entropy(relation_logits, rel_labels.long(), self.cls_weight)
        else:
            loss_relation = self.criterion_loss(relation_logits, rel_labels.long()) # no reweight
        loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())

        # The following code is used to calcaulate sampled attribute loss
        if self.attri_on:
            refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                # have at least one bbox got fg attributes
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(refine_att_logits, attribute_targets, 
                                             fg_bg_sample=self.attribute_sampling, 
                                             bg_fg_ratio=self.attribute_bgfg_ratio)
            return loss_relation, (loss_refine_obj, loss_refine_att)
        else:
            return loss_relation, loss_refine_obj

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)   
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss

class VG_Loss():
    def __init__(self):
        self.pred_weight = torch.FloatTensor([0.0418, 0.1109, 2.1169, 1.3740, 1.1754, 1.2242, 0.4437, 0.4318, 0.1235,
                                              1.2934, 1.2475, 0.8236, 1.8305, 1.1202, 0.9135, 2.1414, 0.4787, 1.4416,
                                              2.2787, 0.4441, 0.0446, 0.1203, 0.0455, 0.3264, 1.1399, 1.3780, 2.2594,
                                              1.7788, 1.8779, 0.0675, 0.0544, 0.0419, 2.2067, 0.4703, 1.3727, 1.5585,
                                              2.0469, 1.1191, 0.4936, 1.8878, 0.2460, 0.3163, 1.6831, 0.2068, 2.1942,
                                              2.4253, 0.9280, 1.2198, 0.0563, 0.2921, 0.0862]).cuda()

        self.iter = 0
        self.head_pred_idx = torch.tensor([41, 7, 22, 49, 23, 8, 21, 43, 20, 48, 1, 40, 50, 31, 30, 29]).cuda()

    def pred_loss(self, rel_labels, rel_logits, alpha):
        if alpha == 1:
            return torch.tensor(0).cuda()

        ### Predicate Curriculum Schedule ###
        self.iter += 1
        weight = self.pred_weight.clone()
        weight[self.head_pred_idx] *= max((1 - self.iter / 30000), 0.2)
        return torch.nn.functional.cross_entropy(rel_logits, rel_labels, weight)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)

        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()



def make_roi_relation_loss_evaluator(cfg):

    loss_evaluator = RelationLossComputation(
        cfg,
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
        cfg.MODEL.USE_REWEIGHT
    )

    return loss_evaluator
