# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.make_layers import make_fc
from .utils_motifs import obj_edge_vectors, encode_box_info, to_onehot, nms_overlaps


class VTransEFeaturePE(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, in_channels, only_vis=False, only_txt=False):
        super(VTransEFeaturePE, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # word embedding
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM

        self.only_vis = only_vis
        self.only_txt = only_txt
        if only_vis == False:
            obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        elif only_vis == True:
            obj_embed_vecs = torch.zeros((self.num_obj_classes, self.embed_dim))

        self.obj_embed = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(obj_embed_vecs, non_blocking=True)

        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        # object & relation context
        self.obj_dim = in_channels
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM

        self.fc_layer = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

        # untreated average features
        self.average_ratio = 0.0005
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS

        if self.effect_analysis:
            self.register_buffer("untreated_obj_feat", torch.zeros(self.obj_dim + self.embed_dim + 128))
            self.register_buffer("untreated_edg_feat", torch.zeros(self.obj_dim + 128))

        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes)
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def forward(self, x, proposals, rel_pair_idxs, logger=None, all_average=False, ctx_average=False):

        if self.only_txt:
            x = torch.zeros_like(x)

        # labels will be used in DecoderRNN during training (for nms)
        obj_dists, obj_preds = self.refine_obj_labels(x, proposals)

        pos_embed = self.pos_embed(encode_box_info(proposals))
        batch_size = x.shape[0]

        # edge level contextual feature
        if (all_average or ctx_average) and self.effect_analysis and (not self.training):
            obj_embed2 = F.softmax(obj_dists, dim=1) @ self.obj_embed2.weight
            obj_rel_rep = cat((self.untreated_edg_feat.view(1, -1).expand(batch_size, -1), obj_embed2), dim=-1)
        else:
            obj_embed2 = self.obj_embed2(obj_preds.long())
            obj_rel_rep = cat((x, pos_embed, obj_embed2), -1)

        edge_ctx = F.relu(self.fc_layer(obj_rel_rep))

        # memorize average feature
        if self.training and self.effect_analysis:
            self.untreated_edg_feat = self.moving_average(self.untreated_edg_feat, cat((x, pos_embed), -1))

        return obj_dists, obj_preds, edge_ctx, None

    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed.weight

        assert proposals[0].mode == 'xyxy'

        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))  # fc

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)  # fc 512 -> 151
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long()
            else:
                obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()

        return obj_dists, obj_preds

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh  # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0  # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds


class VTransEFeature(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super(VTransEFeature, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # word embedding
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(obj_embed_vecs, non_blocking=True)

        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        # object & relation context
        self.obj_dim = in_channels
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM

        self.pred_layer = make_fc(self.obj_dim + self.embed_dim + 128, self.num_obj_classes)
        self.fc_layer = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

        # untreated average features
        self.average_ratio = 0.0005
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS

        if self.effect_analysis:
            self.register_buffer("untreated_obj_feat", torch.zeros(self.obj_dim + self.embed_dim + 128))
            self.register_buffer("untreated_edg_feat", torch.zeros(self.obj_dim + 128))

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def forward(self, x, proposals, rel_pair_idxs, logger=None, all_average=False, ctx_average=False):
        num_objs = [len(b) for b in proposals]
        # labels will be used in DecoderRNN during training (for nms)
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        else:
            obj_labels = None

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(obj_labels.long())
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        assert proposals[0].mode == 'xyxy'
        pos_embed = self.pos_embed(encode_box_info(proposals))

        batch_size = x.shape[0]
        if (all_average or ctx_average) and self.effect_analysis and (not self.training):
            obj_pre_rep = self.untreated_obj_feat.view(1, -1).expand(batch_size, -1)
        else:
            obj_pre_rep = cat((x, obj_embed, pos_embed), -1)

        # object level contextual feature
        obj_dists = self.pred_layer(obj_pre_rep)
        obj_preds = obj_dists.max(-1)[1]
        # edge level contextual feature

        if (all_average or ctx_average) and self.effect_analysis and (not self.training):
            obj_embed2 = F.softmax(obj_dists, dim=1) @ self.obj_embed2.weight
            obj_rel_rep = cat((self.untreated_edg_feat.view(1, -1).expand(batch_size, -1), obj_embed2), dim=-1)
        else:
            obj_embed2 = self.obj_embed2(obj_preds.long())
            obj_rel_rep = cat((x, pos_embed, obj_embed2), -1)

        edge_ctx = F.relu(self.fc_layer(obj_rel_rep))

        # memorize average feature
        if self.training and self.effect_analysis:
            self.untreated_obj_feat = self.moving_average(self.untreated_obj_feat, obj_pre_rep)
            self.untreated_edg_feat = self.moving_average(self.untreated_edg_feat, cat((x, pos_embed), -1))

        return obj_dists, obj_preds, edge_ctx, None