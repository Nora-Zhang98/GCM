# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch, os
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from maskrcnn_benchmark.layers import smooth_l1_loss, kl_div_loss, entropy_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.utils import cat
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature, VTransEFeaturePE
from .model_vctree import VCTreeLSTMContext, VCTreeLSTMContextPE
from .model_motifs import LSTMContextPE, LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import TransformerContext
from .utils_relation import layer_init, get_box_info, get_box_pair_info
from maskrcnn_benchmark.data import get_dataset_statistics
from .utils_motifs import rel_vectors, obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info

from .utils_motifs import to_onehot, encode_box_info
from maskrcnn_benchmark.modeling.make_layers import make_fc


@registry.ROI_RELATION_PREDICTOR.register("PrototypeEmbeddingNetwork")
class PrototypeEmbeddingNetwork(nn.Module):
    def __init__(self, config, in_channels):
        super(PrototypeEmbeddingNetwork, self).__init__()

        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.OIV6_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.OIV6_NUM_CLASSES

        self.cfg = config

        assert in_channels is not None
        self.in_channels = in_channels
        self.obj_dim = in_channels

        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        statistics = get_dataset_statistics(config)

        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.mlp_dim = 2048  # config.MODEL.ROI_RELATION_HEAD.PENET_MLP_DIM
        self.post_emb = nn.Linear(self.obj_dim, self.mlp_dim * 2)

        self.embed_dim = 300  # config.MODEL.ROI_RELATION_HEAD.PENET_EMBED_DIM
        dropout_p = 0.2  # config.MODEL.ROI_RELATION_HEAD.PENET_DROPOUT

        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR,
                                          wv_dim=self.embed_dim)  # load Glove for objects

        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR,
                                     wv_dim=self.embed_dim)  # load Glove for predicates
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)

        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        self.gate_sub = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_obj = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim * 2, self.mlp_dim)

        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.mlp_dim, self.mlp_dim * 2), nn.ReLU(True),
            nn.Dropout(dropout_p), nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        ])

        self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim * 2, 2)

        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_pred = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)

        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)

        self.dropout_sub = nn.Dropout(dropout_p)
        self.dropout_obj = nn.Dropout(dropout_p)
        self.dropout_rel_rep = nn.Dropout(dropout_p)

        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)

        self.down_samp = MLP(self.pooling_dim, self.mlp_dim, self.mlp_dim, 2)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        ##### refine object labels
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.obj_dim = in_channels
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes)
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES
        self.cnt = 0

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):

        add_losses = {}
        add_data = {}

        # refine object labels
        entity_dists, entity_preds = self.refine_obj_labels(roi_features, proposals)
        #####

        entity_rep = self.post_emb(roi_features)  # using the roi features obtained from the faster rcnn
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)

        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)  # xs
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)  # xo

        entity_embeds = self.obj_embed(entity_preds)  # obtaining the word embedding of entities with GloVe

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_preds = entity_preds.split(num_objs, dim=0)
        entity_embeds = entity_embeds.split(num_objs, dim=0)

        fusion_so = []
        pair_preds = []

        for pair_idx, sub_rep, obj_rep, entity_pred, entity_embed, proposal in zip(rel_pair_idxs, sub_reps, obj_reps,
                                                                                   entity_preds, entity_embeds,
                                                                                   proposals):
            s_embed = self.W_sub(entity_embed[pair_idx[:, 0]])  # Ws x ts
            o_embed = self.W_obj(entity_embed[pair_idx[:, 1]])  # Wo x to

            sem_sub = self.vis2sem(sub_rep[pair_idx[:, 0]])  # h(xs)
            sem_obj = self.vis2sem(obj_rep[pair_idx[:, 1]])  # h(xo)

            gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))  # gs
            gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))  # go

            sub = s_embed + sem_sub * gate_sem_sub  # s = Ws x ts + gs · h(xs)  i.e., s = Ws x ts + vs
            obj = o_embed + sem_obj * gate_sem_obj  # o = Wo x to + go · h(xo)  i.e., o = Wo x to + vo

            ##### for the model convergence
            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)
            #####

            fusion_so.append(fusion_func(sub, obj))  # F(s, o)
            pair_preds.append(torch.stack((entity_pred[pair_idx[:, 0]], entity_pred[pair_idx[:, 1]]), dim=1))

        fusion_so = cat(fusion_so, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        sem_pred = self.vis2sem(self.down_samp(union_features))  # h(xu)
        gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))  # gp

        rel_rep = fusion_so - sem_pred * gate_sem_pred  # F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up
        predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes

        ##### for the model convergence
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)

        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))
        ######

        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  # <r_norm, c_norm> / τ
        # the rel_dists will be used to calculate the Le_sim with the ce_loss

        entity_dists = entity_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        if self.training:
            ### Prototype Regularization  ---- cosine similarity
            target_rpredicate_proto_norm = predicate_proto_norm.clone().detach()
            simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (self.num_rel_cls * self.num_rel_cls)
            add_losses.update({"l21_loss": l21})  # Le_sim = ||S||_{2,1}
            ### end

            ### Prototype Regularization  ---- Euclidean distance
            gamma2 = 7.0

            predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1)
            predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(self.num_rel_cls, -1, -1)
            proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(
                dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
            sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
            topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1  # obtain d-, where k2 = 1
            dist_loss = torch.max(torch.zeros(self.num_rel_cls).cuda(),
                                  -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
            add_losses.update({"dist_loss2": dist_loss})
            ### end

            ###  Prototype-based Learning  ---- Euclidean distance
            rel_labels = cat(rel_labels, dim=0)
            gamma1 = 1.0
            rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1)  # r
            predicate_proto_expand = predicate_proto.unsqueeze(dim=0).expand(rel_labels.size(0), -1, -1)  # ci
            distance_set = (rel_rep_expand - predicate_proto_expand).norm(
                dim=2) ** 2  # Distance Set G, gi = ||r-ci||_2^2
            mask_neg = torch.ones(rel_labels.size(0), self.num_rel_cls).cuda()
            mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0
            distance_set_neg = distance_set * mask_neg
            distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]  # gt i.e., g+
            sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)

            topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(
                dim=1) / 10  # obtaining g-, where k1 = 10
            loss_sum = torch.max(torch.zeros(rel_labels.size(0)).cuda(),
                                 distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
            add_losses.update({"loss_dis": loss_sum})  # Le_euc = max(0, (g+) - (g-) + gamma1)
            ### end
        else:
            if self.cfg.MODEL.TRAIN_INFER:
                cat_labels = cat(rel_labels, dim=0)
                self.one_epoch(rel_rep_norm, cat_labels)

        return entity_dists, rel_dists, add_losses

    def one_epoch(self, rel_rep_norm, label):  # ONLY restore foreground samples when TRAIN_INFER is True
        self.cnt += 1
        fg_mask = label != 0
        flip_path = "infer_train_feat"
        path = os.path.join(self.cfg.OUTPUT_DIR, flip_path)
        os.makedirs(path, exist_ok=True)
        data = {
            'cnt': self.cnt,
            'rel_rep_norm': rel_rep_norm[fg_mask],
            'fg_label': label[fg_mask]
        }
        torch.save(data, os.path.join(path, "{}.pkl".format(self.cnt)))

    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        # pos_embed = self.pos_embed(encode_box_info(proposals))

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        assert proposals[0].mode == 'xyxy'

        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)  # 512 -> 151
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


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05, imb_factor=1):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.imb_factor = imb_factor

    @torch.no_grad()
    def iterate(self, Q):
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / Q.shape[1]

        # obtain permutation/order from the marginals
        marginals_argsort = torch.argsort(Q.sum(1))
        marginals_argsort = marginals_argsort.detach()
        r = []
        for i in range(Q.shape[0]):
            r.append((1 / self.imb_factor) ** (i / (Q.shape[0] - 1.0)))

        r = np.array(r)
        r = r * (Q.shape[1] / Q.shape[0])  # Per-class distribution in the mini-batch
        r = torch.from_numpy(r).cuda(non_blocking=True)
        r[marginals_argsort] = torch.sort(r)[0]  # Sort/permute based on the data order
        r = torch.clamp(r, min=1)  # Clamp the min to have a balance distribution for the tail classes
        r /= r.sum()  # Scaling to make it prob
        for it in range(self.num_iters):
            u = torch.sum(Q, dim=1)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    @torch.no_grad()
    def forward(self, logits, prior_P=None):
        # get assignments
        # import pdb; pdb.set_trace()
        B, K = logits.shape
        if prior_P is not None:
            denominator = self.epsilon + self.epsilon2
            prior_withreg = - torch.log(prior_P / B) * self.epsilon2
            q = (logits + prior_withreg) / denominator
        else:
            q = logits / self.epsilon

        M = torch.max(q)
        q -= M

        q = torch.exp(q).t()
        return self.iterate(q)


@registry.ROI_RELATION_PREDICTOR.register("PrototypeEmbeddingNetworkGCM")
class PrototypeEmbeddingNetworkGCM(nn.Module):
    def __init__(self, config, in_channels, only_vis=False, only_txt=False):
        super(PrototypeEmbeddingNetworkGCM, self).__init__()
        self.cfg = config

        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.OIV6_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.OIV6_NUM_CLASSES

        assert in_channels is not None
        self.in_channels = in_channels
        self.obj_dim = in_channels

        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        statistics = get_dataset_statistics(config)

        obj_classes, rel_classes= statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)

        self.aug_num_rel_cls = int((self.num_rel_cls-1) * 1.1)
        self.num_bg_cls = self.aug_num_rel_cls - (self.num_rel_cls - 1) 

        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.mlp_dim = 2048  # config.MODEL.ROI_RELATION_HEAD.PENET_MLP_DIM
        self.post_emb = nn.Linear(self.obj_dim, self.mlp_dim * 2)

        self.embed_dim = 300  # config.MODEL.ROI_RELATION_HEAD.PENET_EMBED_DIM
        dropout_p = 0.2  # config.MODEL.ROI_RELATION_HEAD.PENET_DROPOUT

        self.only_vis = only_vis
        self.only_txt = only_txt
        if only_vis == False:
            obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR,
                                              wv_dim=self.embed_dim) 
        elif only_vis == True:
            obj_embed_vecs = torch.zeros((self.num_obj_classes, self.embed_dim))

        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR,
                                     wv_dim=self.embed_dim) 
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)

        self.prefix_dim = 300  

        # CoT generated semantic group, for VG
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            action_list = ['wearing', 'holding', 'sitting on', 'wears', 'riding', 'standing on', 'carrying', 'walking on',
                           'eating', 'using', 'laying on', 'playing', 'flying in']
            position_list = ['on', 'in', 'near', 'with', 'behind', 'above', 'under', 'in front of', 'at', 'attached to',
                             'over', 'between', 'along', 'across', 'against', 'on back of', 'lying on', 'walking in']
            possession_list = ['has', 'belonging to', 'for', 'part of', 'from']
            state_list = ['covering', 'covered in', 'mounted on', 'painted on', 'made of', 'growing on']
            descr_list = ['looking at', 'watching']
            conne_list = ['of', 'to', 'and']
            unclear_list = ['hanging from', 'parked on', 'says']

            # avg embedding within the same semantic group as the semantic group knowledge
            prefix_vecs = torch.randn([8, self.prefix_dim])
            prefix_vecs[1] = obj_edge_vectors(action_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[2] = obj_edge_vectors(position_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[3] = obj_edge_vectors(possession_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[4] = obj_edge_vectors(state_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[5] = obj_edge_vectors(descr_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[6] = obj_edge_vectors(conne_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[7] = obj_edge_vectors(unclear_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)

            action_ids = [rel_classes.index(i) for i in action_list]
            position_ids = [rel_classes.index(i) for i in position_list]
            possession_ids = [rel_classes.index(i) for i in possession_list]
            state_ids = [rel_classes.index(i) for i in state_list]
            descr_ids = [rel_classes.index(i) for i in descr_list]
            conne_ids = [rel_classes.index(i) for i in conne_list]
            unclear_ids = [rel_classes.index(i) for i in unclear_list]

            # predicates within the same semantic group have same knowledge
            prefix_embed = torch.Tensor(self.num_rel_cls, self.prefix_dim)
            prefix_embed.normal_(0, 1)
            prefix_embed[action_ids] = prefix_vecs[1]
            prefix_embed[position_ids] = prefix_vecs[2]
            prefix_embed[possession_ids] = prefix_vecs[3]
            prefix_embed[state_ids] = prefix_vecs[4]
            prefix_embed[descr_ids] = prefix_vecs[5]
            prefix_embed[conne_ids] = prefix_vecs[6]
            prefix_embed[unclear_ids] = prefix_vecs[7]

        # CoT generated semantic group, for GQA
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            action_list = ['wearing', 'holding', 'sitting on', 'riding', 'carrying', 'walking on', 'lying on', 'eating',
                           'watching', 'hanging on', 'using', 'playing with', 'throwing', 'catching', 'grazing on', 'cutting', 'feeding',
                           'running on', 'talking to', 'pulling', 'reaching for', 'skiing on', 'hitting', 'sitting at', 'leaning on',
                           'standing on', 'touching', 'swinging', 'talking on', 'pulled by', 'hang on']
            position_list = ['on', 'in', 'under', 'behind', 'in front of', 'next to', 'above', 'below', 'by', 'with',
                             'at', 'around', 'on the front of', 'on the side of', 'on the bottom of', 'beneath',
                             'crossing', 'standing next to', 'standing near', 'standing behind',
                             'surrounded by', 'standing by', 'standing in front of', 'close to', 'sitting next to',
                             'near', 'beside', 'on the back of', 'reflected in', 'facing',
                             'on top of', 'standing in', 'sitting in', 'lying on', 'parked along']
            state_list = ['covered by', 'covered in', 'filled with', 'mounted on', 'resting on', 'printed on',
                          'floating in', 'attached to', 'contain', 'surrounding', 'full of', 'hanging from',
                          'leaning against', 'covering', 'covered with', 'growing on', 'growing in', 'worn on']
            movement_list = ['flying in', 'flying', 'walking in', 'walking down', 'walking with', 'driving',
                             'driving on', 'driving down', 'swimming in', 'grazing in', 'parked on']
            inter_list = ['looking at', 'playing on', 'playing in', 'waiting for']
            unclear_list = ['of']

            # avg embedding within the same semantic group as the semantic group knowledge
            prefix_vecs = torch.randn([7, self.prefix_dim])
            prefix_vecs[1] = obj_edge_vectors(action_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[2] = obj_edge_vectors(position_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[3] = obj_edge_vectors(state_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[4] = obj_edge_vectors(movement_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[5] = obj_edge_vectors(inter_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[6] = obj_edge_vectors(unclear_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)

            action_ids = [rel_classes.index(i) for i in action_list]
            position_ids = [rel_classes.index(i) for i in position_list]
            state_ids = [rel_classes.index(i) for i in state_list]
            movement_ids = [rel_classes.index(i) for i in movement_list]
            inter_ids = [rel_classes.index(i) for i in inter_list]
            unclear_ids = [rel_classes.index(i) for i in unclear_list]

            # predicates within the same semantic group have same knowledge
            prefix_embed = torch.Tensor(self.num_rel_cls, self.prefix_dim)
            prefix_embed.normal_(0, 1)
            prefix_embed[action_ids] = prefix_vecs[1]
            prefix_embed[position_ids] = prefix_vecs[2]
            prefix_embed[state_ids] = prefix_vecs[3]
            prefix_embed[movement_ids] = prefix_vecs[4]
            prefix_embed[inter_ids] = prefix_vecs[5]
            prefix_embed[unclear_ids] = prefix_vecs[6]

        # CoT generated semantic group, for OIV6
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
            action_list = ['holds', 'wears', 'surf', 'hang', 'drink', 'holding_hands', 'ride', 'dance', 'skateboard',
                           'catch', 'highfive', 'eat', 'cut', 'handshake', 'kiss', 'talk_on_phone', 'throw', 'hits',
                           'kick', 'ski', 'plays', 'read', 'snowboard']
            position_list = ['at', 'on', 'under', 'inside_of']
            inter_list = ['interacts_with', 'hug']
            contain_list = ['contain']

            # avg embedding within the same semantic group as the semantic group knowledge
            prefix_vecs = torch.randn([5, self.prefix_dim])
            prefix_vecs[1] = obj_edge_vectors(action_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[2] = obj_edge_vectors(position_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[3] = obj_edge_vectors(inter_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[4] = obj_edge_vectors(contain_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)

            action_ids = [rel_classes.index(i) for i in action_list]
            position_ids = [rel_classes.index(i) for i in position_list]
            inter_ids = [rel_classes.index(i) for i in inter_list]
            contain_ids = [rel_classes.index(i) for i in contain_list]

            # predicates within the same semantic group have same knowledge
            prefix_embed = torch.Tensor(self.num_rel_cls, self.prefix_dim)
            prefix_embed.normal_(0, 1)
            prefix_embed[action_ids] = prefix_vecs[1]
            prefix_embed[position_ids] = prefix_vecs[2]
            prefix_embed[inter_ids] = prefix_vecs[3]
            prefix_embed[contain_ids] = prefix_vecs[4]

        self.prefix_vecs = prefix_vecs
        prefix_bg = torch.Tensor(self.num_bg_cls - 1, self.prefix_dim) 
        prefix_bg.normal_(0, 1)
        prefix_embed = torch.cat([prefix_bg, prefix_embed], dim=0) 
        self.prefix_embed = torch.nn.Parameter(prefix_embed)

        self.rel_embed = nn.Embedding(self.aug_num_rel_cls, self.embed_dim) 
        bg_rel_embed_vecs = torch.Tensor(self.num_bg_cls - 1, self.embed_dim) 
        bg_rel_embed_vecs.normal_(0, 1)
        rel_embed_vecs = torch.cat([bg_rel_embed_vecs, rel_embed_vecs], dim=0)
        self.rel_embed = nn.Embedding(self.aug_num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)

        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2) 

        self.gate_sub = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_obj = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim * 2, self.mlp_dim)

        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.mlp_dim, self.mlp_dim * 2), nn.ReLU(True),
            nn.Dropout(dropout_p), nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        ])

        self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim * 2, 2)

        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_pred = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)

        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)

        self.dropout_sub = nn.Dropout(dropout_p)
        self.dropout_obj = nn.Dropout(dropout_p)
        self.dropout_rel_rep = nn.Dropout(dropout_p)

        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)

        self.down_samp = MLP(self.pooling_dim, self.mlp_dim, self.mlp_dim, 2)

        file_path = '/all_mean_var_'
        if self.cfg.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            file_path += 'vg_'
        elif self.cfg.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            file_path += 'gqa_'
        elif self.cfg.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
            file_path += 'oiv6_'

        file_path += 'penet.pkl'
        class_var = torch.load(self.cfg.OUTPUT_DIR + file_path, map_location=torch.device('cpu'))['all_var']

        class_var = F.normalize(class_var, p=2, dim=0)
        logit_tao = torch.cat([torch.ones([self.num_bg_cls])*0.07, class_var.sum(dim=1) / 10000], dim=0)
        self.logit_scale2 = nn.Parameter(torch.ones([]) * np.log(1 / logit_tao))
        ##### refine object labels
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.obj_dim = in_channels
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes)
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES
        self.uni_dist = (torch.ones(self.num_bg_cls) / self.num_bg_cls).cuda() 
        self.cnt = 0

        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_sem_grp = 8
            self.sem_labels = torch.zeros(len(rel_classes)).cuda()
            self.sem_labels[action_ids] = 1
            self.sem_labels[position_ids] = 2
            self.sem_labels[possession_ids] = 3
            self.sem_labels[state_ids] = 4
            self.sem_labels[descr_ids] = 5
            self.sem_labels[conne_ids] = 6
            self.sem_labels[unclear_ids] = 7
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_sem_grp = 7
            self.sem_labels = torch.zeros(len(rel_classes)).cuda()
            self.sem_labels[action_ids] = 1
            self.sem_labels[position_ids] = 2
            self.sem_labels[state_ids] = 3
            self.sem_labels[movement_ids] = 4
            self.sem_labels[inter_ids] = 5
            self.sem_labels[unclear_ids] = 6
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
            self.num_sem_grp = 5
            self.sem_labels = torch.zeros(len(rel_classes)).cuda()
            self.sem_labels[action_ids] = 1
            self.sem_labels[position_ids] = 2
            self.sem_labels[inter_ids] = 3
            self.sem_labels[contain_ids] = 4

        self.sup_linear = nn.utils.weight_norm(nn.Linear(self.mlp_dim * 2, self.num_sem_grp, bias=False))
        self.sup_linear.weight_g.data.fill_(1)  
        self.sup_linear.weight_g.requires_grad = False
        self.sinkhorn = SinkhornKnopp()
        self.sup_compress = nn.Linear(self.prefix_dim + self.mlp_dim * 2, self.mlp_dim * 2)
        self.sup_loss = nn.CrossEntropyLoss()
        self.norm_sup = nn.LayerNorm(self.prefix_dim + self.mlp_dim * 2)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features,
                logger=None):

        add_losses = {}
        add_data = {}

        # refine object labels
        entity_dists, entity_preds = self.refine_obj_labels(roi_features, proposals)
        #####

        if self.only_txt:
            roi_features = torch.zeros_like(roi_features)
            union_features = torch.zeros_like(union_features)

        entity_rep = self.post_emb(roi_features)  # using the roi features obtained from the faster rcnn
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)

        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)  # xs
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)  # xo

        entity_embeds = self.obj_embed(entity_preds)  # obtaining the word embedding of entities with GloVe
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_preds = entity_preds.split(num_objs, dim=0)
        entity_embeds = entity_embeds.split(num_objs, dim=0)

        fusion_so = []
        pair_preds = []

        for pair_idx, sub_rep, obj_rep, entity_pred, entity_embed, proposal in zip(rel_pair_idxs, sub_reps, obj_reps,
                                                                                   entity_preds, entity_embeds,
                                                                                   proposals):
            s_embed = self.W_sub(entity_embed[pair_idx[:, 0]])  # Ws x ts
            o_embed = self.W_obj(entity_embed[pair_idx[:, 1]])  # Wo x to

            sem_sub = self.vis2sem(sub_rep[pair_idx[:, 0]])  # h(xs)
            sem_obj = self.vis2sem(obj_rep[pair_idx[:, 1]])  # h(xo)

            gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))  # gs
            gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))  # go

            sub = s_embed + sem_sub * gate_sem_sub  # s = Ws x ts + gs · h(xs)  i.e., s = Ws x ts + vs
            obj = o_embed + sem_obj * gate_sem_obj  # o = Wo x to + go · h(xo)  i.e., o = Wo x to + vo

            ##### for the model convergence
            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)
            #####

            fusion_so.append(fusion_func(sub, obj))  # F(s, o)
            pair_preds.append(torch.stack((entity_pred[pair_idx[:, 0]], entity_pred[pair_idx[:, 1]]), dim=1))

        fusion_so = cat(fusion_so, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        sem_pred = self.vis2sem(self.down_samp(union_features))  # h(xu)
        gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))  # gp

        rel_rep = fusion_so - sem_pred * gate_sem_pred  # F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up

        predicate_proto = self.W_pred(self.prefix_embed*0.2 + self.rel_embed.weight)

        ##### for the model convergence
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)
        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))
        ######

        sup_dist = self.sup_linear(F.normalize(rel_rep, dim=1))
        sup_dist_logits = self.sinkhorn(sup_dist)

        if self.training:
            sem_labels = self.sem_labels[cat(rel_labels)]
            sup_cls_loss = self.sup_loss(sup_dist_logits, sem_labels.long())
            add_losses.update({'sup_cls_loss': sup_cls_loss * 0.1})

        sup_rep = sup_dist_logits @ self.prefix_vecs.cuda()
        rel_rep = self.sup_compress(self.norm_sup(torch.cat([sup_rep, rel_rep], dim=1)))

        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale2.exp()  # <r_norm, c_norm> / τ
        # the rel_dists will be used to calculate the Le_sim with the ce_loss

        if self.training and not self.only_vis and not self.only_txt:

            ### Prototype Regularization  ---- cosine similarity
            target_rpredicate_proto_norm = predicate_proto_norm.clone().detach()
            simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (
                        self.aug_num_rel_cls * self.aug_num_rel_cls)
            add_losses.update({"l21_loss": l21})  # Le_sim = ||S||_{2,1}
            ### end

            ### Prototype Regularization  ---- Euclidean distance
            gamma2 = 7.0
            predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, self.aug_num_rel_cls, -1)
            predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(self.aug_num_rel_cls, -1,
                                                                                 -1)
            proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(
                dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
            sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
            topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1  # obtain d-, where k2 = 1
            dist_loss = torch.max(torch.zeros(self.aug_num_rel_cls).cuda(),
                                  -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
            add_losses.update({"dist_loss2": dist_loss})
            ### end

            ###  Prototype-based Learning  ---- Euclidean distance
            rel_labels = cat(rel_labels, dim=0)
            gamma1 = 1.0
            rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, self.aug_num_rel_cls, -1)  # r
            predicate_proto_expand = predicate_proto.unsqueeze(dim=0).expand(rel_labels.size(0), -1, -1)  # ci
            distance_set = (rel_rep_expand - predicate_proto_expand).norm(
                dim=2) ** 2  # Distance Set G, gi = ||r-ci||_2^2
            mask_neg = torch.ones(rel_labels.size(0), self.aug_num_rel_cls).cuda()

            fg_id = rel_labels.nonzero().view(-1)
            fg_rel_labels = rel_labels[fg_id] + self.num_bg_cls - 1
            bg_id = (rel_labels == 0).nonzero().view(-1)
            num_bg = len(bg_id)
            bg_rel_labels = torch.randint(1, self.num_bg_cls, (num_bg,)).cuda()
            aug_rel_labels = torch.zeros_like(rel_labels)
            aug_rel_labels[fg_id] = fg_rel_labels
            aug_rel_labels[bg_id] = bg_rel_labels

            mask_neg[torch.arange(aug_rel_labels.size(0)), aug_rel_labels] = 0
            distance_set_neg = distance_set * mask_neg
            distance_set_pos = distance_set[torch.arange(aug_rel_labels.size(0)), aug_rel_labels]
            sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
            topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(
                dim=1) / 10  # obtaining g-, where k1 = 10
            loss_sum = torch.max(torch.zeros(aug_rel_labels.size(0)).cuda(),
                                 distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
            add_losses.update({"loss_dis": loss_sum})  # Le_euc = max(0, (g+) - (g-) + gamma1)

            bg_rel_dists = rel_dists[bg_id][:, :self.num_bg_cls]

            bg_rel_dists = (bg_rel_dists - bg_rel_dists.mean(dim=1).reshape(-1, 1)) / bg_rel_dists.std(dim=1).reshape(
                -1, 1)
            kl_loss = 10 * F.kl_div(bg_rel_dists.softmax(dim=-1).mean(dim=0).log(), self.uni_dist,
                                    reduction='mean')
            add_losses.update({"kl_loss": kl_loss})
            ### end

        bg_dists = rel_dists[:, :self.num_bg_cls]
        fg_dists = rel_dists[:, self.num_bg_cls:]
        rel_dists = torch.cat([bg_dists, fg_dists], dim=1)

        rel_dists = rel_dists[:, self.num_bg_cls - 1:]
        entity_dists = entity_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        return entity_dists, rel_dists, add_losses

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input
        return holder

    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        pos_embed = self.pos_embed(encode_box_info(proposals))

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        assert proposals[0].mode == 'xyxy'

        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)  # 512 -> 151
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

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def fusion_func(x, y):
    return F.relu(x + y) - (x - y) ** 2


@registry.ROI_RELATION_PREDICTOR.register("TransformerPredictor")
class TransformerPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TransformerPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.OIV6_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.OIV6_NUM_CLASSES

        self.cfg = config
        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # module construct
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

        self.cnt = 0

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        if not self.training and self.cfg.MODEL.TRAIN_INFER:
            cat_labels = cat(rel_labels, dim=0)
            self.one_epoch((visual_rep + prod_rep) / 2, cat_labels)

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)

        # use frequence bias
        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred)

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses

    def one_epoch(self, rel_rep_norm, label):
        self.cnt += 1
        fg_mask = label != 0
        flip_path = "infer_train_feat"
        path = os.path.join(self.cfg.OUTPUT_DIR, flip_path)
        os.makedirs(path, exist_ok=True)
        data = {
            'cnt': self.cnt,
            'rel_rep_norm': rel_rep_norm[fg_mask],
            'fg_label': label[fg_mask]
        }
        torch.save(data, os.path.join(path, "{}.pkl".format(self.cnt)))


@registry.ROI_RELATION_PREDICTOR.register("IMPPredictor")
class IMPPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(IMPPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = False

        assert in_channels is not None

        self.context_layer = IMPContext(config, self.num_obj_cls, self.num_rel_cls, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # freq
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        # encode context infomation
        obj_dists, rel_dists = self.context_layer(roi_features, proposals, union_features, rel_pair_idxs, logger)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        if self.use_bias:
            obj_preds = obj_dists.max(-1)[1]
            obj_preds = obj_preds.split(num_objs, dim=0)

            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
                pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
            pair_pred = cat(pair_preds, dim=0)

            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("VTransEPredictor")
class VTransEPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(VTransEPredictor, self).__init__()
        self.cfg = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.OIV6_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.OIV6_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)

        self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)
        self.criterion_loss = nn.CrossEntropyLoss()
        self.cnt = 0

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        if not self.training and self.cfg.MODEL.TRAIN_INFER:
            cat_labels = cat(rel_labels, dim=0)
            self.one_epoch(prod_rep, cat_labels)

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        add_losses = {}
        return obj_dists, rel_dists, add_losses

    def one_epoch(self, rel_rep_norm, label):  # ONLY restore foreground samples when TRAIN_INFER is True
        self.cnt += 1
        fg_mask = label != 0
        flip_path = "infer_train_feat"
        path = os.path.join(self.cfg.OUTPUT_DIR, flip_path)
        os.makedirs(path, exist_ok=True)
        data = {
            'cnt': self.cnt,
            'rel_rep_norm': rel_rep_norm[fg_mask],
            'fg_label': label[fg_mask]
        }
        torch.save(data, os.path.join(path, "{}.pkl".format(self.cnt)))


@registry.ROI_RELATION_PREDICTOR.register("VTransEPredictorGCM")
class VTransEPredictorGCM(nn.Module):
    def __init__(self, config, in_channels, only_vis=False, only_txt=False):
        super(VTransEPredictorGCM, self).__init__()
        self.cfg = config
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.OIV6_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.OIV6_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)

        # modality erasing
        self.only_vis = only_vis
        self.only_txt = only_txt

        self.context_layer = VTransEFeaturePE(config, obj_classes, rel_classes, in_channels, self.only_vis, self.only_txt)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        # layer_init(self.rel_compress, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)
        self.criterion_loss = nn.CrossEntropyLoss()

        self.aug_num_rel_cls = int((len(rel_classes) - 1) * 1.1)         # augmented class number, α=0.1
        self.num_bg_cls = self.aug_num_rel_cls - (self.num_rel_cls - 1)  # bg class number
        self.uni_dist = (torch.ones(self.num_bg_cls) / self.num_bg_cls).cuda()

        self.embed_dim = 300
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)
        self.prefix_dim = 300

        # CoT generated semantic group, for VG
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            action_list = ['wearing', 'holding', 'sitting on', 'wears', 'riding', 'standing on', 'carrying',
                           'walking on', 'eating', 'using', 'laying on', 'playing', 'flying in']
            position_list = ['on', 'in', 'near', 'with', 'behind', 'above', 'under', 'in front of', 'at', 'attached to',
                             'over', 'between', 'along', 'across', 'against', 'on back of', 'lying on', 'walking in']
            possession_list = ['has', 'belonging to', 'for', 'part of', 'from']
            state_list = ['covering', 'covered in', 'mounted on', 'painted on', 'made of', 'growing on']
            descr_list = ['looking at', 'watching']
            conne_list = ['of', 'to', 'and']
            unclear_list = ['hanging from', 'parked on', 'says']

            # avg embedding within the same semantic group as the semantic group knowledge
            prefix_vecs = torch.randn([8, self.prefix_dim])
            prefix_vecs[1] = obj_edge_vectors(action_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[2] = obj_edge_vectors(position_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[3] = obj_edge_vectors(possession_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[4] = obj_edge_vectors(state_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[5] = obj_edge_vectors(descr_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[6] = obj_edge_vectors(conne_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[7] = obj_edge_vectors(unclear_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)

            action_ids = [rel_classes.index(i) for i in action_list]
            position_ids = [rel_classes.index(i) for i in position_list]
            possession_ids = [rel_classes.index(i) for i in possession_list]
            state_ids = [rel_classes.index(i) for i in state_list]
            descr_ids = [rel_classes.index(i) for i in descr_list]
            conne_ids = [rel_classes.index(i) for i in conne_list]
            unclear_ids = [rel_classes.index(i) for i in unclear_list]

            # predicates within the same semantic group have same knowledge
            prefix_embed = torch.Tensor(self.num_rel_cls, self.prefix_dim)
            prefix_embed.normal_(0, 1)
            prefix_embed[action_ids] = prefix_vecs[1]
            prefix_embed[position_ids] = prefix_vecs[2]
            prefix_embed[possession_ids] = prefix_vecs[3]
            prefix_embed[state_ids] = prefix_vecs[4]
            prefix_embed[descr_ids] = prefix_vecs[5]
            prefix_embed[conne_ids] = prefix_vecs[6]
            prefix_embed[unclear_ids] = prefix_vecs[7]

        # CoT generated semantic group, for GQA
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            action_list = ['wearing', 'holding', 'sitting on', 'riding', 'carrying', 'walking on', 'lying on', 'eating',
                           'watching', 'hanging on', 'using', 'playing with', 'throwing', 'catching', 'grazing on',
                           'cutting', 'feeding', 'running on', 'talking to', 'pulling', 'reaching for', 'skiing on',
                           'hitting', 'sitting at', 'leaning on', 'standing on', 'touching', 'swinging', 'talking on',
                           'pulled by', 'hang on']
            position_list = ['on', 'in', 'under', 'behind', 'in front of', 'next to', 'above', 'below', 'by', 'with',
                             'at', 'around', 'on the front of', 'on the side of', 'on the bottom of', 'beneath',
                             'crossing', 'standing next to', 'standing near', 'standing behind',
                             'surrounded by', 'standing by', 'standing in front of', 'close to', 'sitting next to',
                             'near', 'beside', 'on the back of', 'reflected in', 'facing',
                             'on top of', 'standing in', 'sitting in', 'lying on', 'parked along']
            state_list = ['covered by', 'covered in', 'filled with', 'mounted on', 'resting on', 'printed on',
                          'floating in', 'attached to', 'contain', 'surrounding', 'full of', 'hanging from',
                          'leaning against', 'covering', 'covered with', 'growing on', 'growing in', 'worn on']
            movement_list = ['flying in', 'flying', 'walking in', 'walking down', 'walking with', 'driving',
                             'driving on', 'driving down', 'swimming in', 'grazing in', 'parked on']
            inter_list = ['looking at', 'playing on', 'playing in', 'waiting for']
            unclear_list = ['of']

            # avg embedding within the same semantic group as the semantic group knowledge
            prefix_vecs = torch.randn([7, self.prefix_dim])
            prefix_vecs[1] = obj_edge_vectors(action_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[2] = obj_edge_vectors(position_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[3] = obj_edge_vectors(state_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[4] = obj_edge_vectors(movement_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[5] = obj_edge_vectors(inter_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[6] = obj_edge_vectors(unclear_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)

            action_ids = [rel_classes.index(i) for i in action_list]
            position_ids = [rel_classes.index(i) for i in position_list]
            state_ids = [rel_classes.index(i) for i in state_list]
            movement_ids = [rel_classes.index(i) for i in movement_list]
            inter_ids = [rel_classes.index(i) for i in inter_list]
            unclear_ids = [rel_classes.index(i) for i in unclear_list]

            # predicates within the same semantic group have same knowledge
            prefix_embed = torch.Tensor(self.num_rel_cls, self.prefix_dim)
            prefix_embed.normal_(0, 1)
            prefix_embed[action_ids] = prefix_vecs[1]
            prefix_embed[position_ids] = prefix_vecs[2]
            prefix_embed[state_ids] = prefix_vecs[3]
            prefix_embed[movement_ids] = prefix_vecs[4]
            prefix_embed[inter_ids] = prefix_vecs[5]
            prefix_embed[unclear_ids] = prefix_vecs[6]

        # CoT generated semantic group, for OIV6
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
            action_list = ['holds', 'wears', 'surf', 'hang', 'drink', 'holding_hands', 'ride', 'dance', 'skateboard',
                           'catch', 'highfive', 'eat', 'cut', 'handshake', 'kiss', 'talk_on_phone', 'throw',
                           'hits', 'kick', 'ski', 'plays', 'read', 'snowboard']
            position_list = ['at', 'on', 'under', 'inside_of']
            inter_list = ['interacts_with', 'hug']
            contain_list = ['contain']

            # avg embedding within the same semantic group as the semantic group knowledge
            prefix_vecs = torch.randn([5, self.prefix_dim])
            prefix_vecs[1] = obj_edge_vectors(action_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[2] = obj_edge_vectors(position_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[3] = obj_edge_vectors(inter_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[4] = obj_edge_vectors(contain_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)

            action_ids = [rel_classes.index(i) for i in action_list]
            position_ids = [rel_classes.index(i) for i in position_list]
            inter_ids = [rel_classes.index(i) for i in inter_list]
            contain_ids = [rel_classes.index(i) for i in contain_list]

            # predicates within the same semantic group have same knowledge
            prefix_embed = torch.Tensor(self.num_rel_cls, self.prefix_dim)
            prefix_embed.normal_(0, 1)
            prefix_embed[action_ids] = prefix_vecs[1]
            prefix_embed[position_ids] = prefix_vecs[2]
            prefix_embed[inter_ids] = prefix_vecs[3]
            prefix_embed[contain_ids] = prefix_vecs[4]

        self.rel_embed = nn.Embedding(self.aug_num_rel_cls, self.embed_dim)
        bg_rel_embed_vecs = torch.Tensor(self.num_bg_cls - 1, self.embed_dim)           # bg prototypes
        bg_rel_embed_vecs.normal_(0, 1)
        rel_embed_vecs = torch.cat([bg_rel_embed_vecs, rel_embed_vecs], dim=0)  # augmented prototypes
        self.rel_embed = nn.Embedding(self.aug_num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)

        self.prefix_vecs = prefix_vecs
        prefix_bg = torch.Tensor(self.num_bg_cls - 1, self.prefix_dim)
        prefix_bg.normal_(0, 1)
        prefix_embed = torch.cat([prefix_bg, prefix_embed], dim=0)
        self.prefix_embed = torch.nn.Parameter(prefix_embed)

        self.W_pred = MLP(self.embed_dim, self.pooling_dim, self.pooling_dim, 2)     # fusion of semantic group rep and rel rep
        self.project_head = MLP(self.pooling_dim, self.pooling_dim, self.pooling_dim, 2)
        dropout_p = 0.2
        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)

        file_path = '/all_mean_var_'
        if self.cfg.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            file_path += 'vg_'
        elif self.cfg.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            file_path += 'gqa_'
        elif self.cfg.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
            file_path += 'oiv6_'

        file_path += 'vtranse.pkl'

        class_var = torch.load(self.cfg.OUTPUT_DIR + file_path, map_location=torch.device('cpu'))['all_var']
        class_var = F.normalize(class_var, p=2, dim=0)
        logit_tao = torch.cat([torch.ones([self.num_bg_cls]) * 0.07, class_var.sum(dim=1) / 10000], dim=0)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / logit_tao))

        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_sem_grp = 8
            self.sem_labels = torch.zeros(len(rel_classes)).cuda()
            self.sem_labels[action_ids] = 1
            self.sem_labels[position_ids] = 2
            self.sem_labels[possession_ids] = 3
            self.sem_labels[state_ids] = 4
            self.sem_labels[descr_ids] = 5
            self.sem_labels[conne_ids] = 6
            self.sem_labels[unclear_ids] = 7
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_sem_grp = 7
            self.sem_labels = torch.zeros(len(rel_classes)).cuda()
            self.sem_labels[action_ids] = 1
            self.sem_labels[position_ids] = 2
            self.sem_labels[state_ids] = 3
            self.sem_labels[movement_ids] = 4
            self.sem_labels[inter_ids] = 5
            self.sem_labels[unclear_ids] = 6
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
            self.num_sem_grp = 5
            self.sem_labels = torch.zeros(len(rel_classes)).cuda()
            self.sem_labels[action_ids] = 1
            self.sem_labels[position_ids] = 2
            self.sem_labels[inter_ids] = 3
            self.sem_labels[contain_ids] = 4

        # initialization of semantic group classifier
        self.sup_linear = nn.utils.weight_norm(nn.Linear(self.pooling_dim, self.num_sem_grp, bias=False))
        self.sup_linear.weight_g.data.fill_(1) 
        self.sup_linear.weight_g.requires_grad = False

        self.sinkhorn = SinkhornKnopp()         # initialization of IGLR
        self.sup_compress = nn.Linear(self.prefix_dim + self.pooling_dim, self.pooling_dim)
        self.sup_loss = nn.CrossEntropyLoss()   # for semantic group classification
        self.norm_sup = nn.LayerNorm(self.prefix_dim + self.pooling_dim)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        add_losses = {}
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        # inject semantic group knowledge into rel rep
        predicate_proto = self.W_pred(self.prefix_embed * 0.2 + self.rel_embed.weight)

        prod_rep = self.project_head(self.dropout_rel(torch.relu(prod_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))

        # semantic group classification
        sup_dist = self.sup_linear(F.normalize(prod_rep, dim=1))
        sup_dist_logits = self.sinkhorn(sup_dist)   # IGLR for regularization

        if self.training:
            sem_labels = self.sem_labels[cat(rel_labels)]
            sup_cls_loss = self.sup_loss(sup_dist_logits, sem_labels.long())
            add_losses.update({'sup_cls_loss': sup_cls_loss * 0.1})

        sup_rep = sup_dist_logits @ self.prefix_vecs.cuda()
        prod_rep = self.sup_compress(self.norm_sup(torch.cat([sup_rep, prod_rep], dim=1)))
        prod_rep_norm = prod_rep / prod_rep.norm(dim=1, keepdim=True)  # r_norm
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm  (cosine similarity) & (Relation Prediction)
        rel_dists = prod_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  # <r_norm, c_norm> / τ

        if self.training and not self.only_vis and not self.only_txt:
            rel_labels = cat(rel_labels, dim=0)
            # separate fg and bg samples
            fg_id = rel_labels.nonzero().view(-1)
            fg_rel_labels = rel_labels[fg_id] + self.num_bg_cls - 1
            bg_id = (rel_labels == 0).nonzero().view(-1)
            num_bg = len(bg_id)
            bg_rel_labels = torch.randint(1, self.num_bg_cls, (num_bg,)).cuda()
            aug_rel_labels = torch.zeros_like(rel_labels)
            aug_rel_labels[fg_id] = fg_rel_labels
            aug_rel_labels[bg_id] = bg_rel_labels
            # uniform constraint for bg dists
            bg_rel_dists = rel_dists[bg_id][:, :self.num_bg_cls]  # reserve bg dists
            bg_rel_dists = (bg_rel_dists - bg_rel_dists.mean(dim=1).reshape(-1, 1)) / bg_rel_dists.std(dim=1).reshape(-1, 1)
            kl_loss = 10 * F.kl_div(bg_rel_dists.softmax(dim=-1).mean(dim=0).log(), self.uni_dist, reduction='mean')
            add_losses.update({"kl_loss": kl_loss})

        rel_dists = rel_dists[:, self.num_bg_cls - 1:]   # slice operation to return to the original class number

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor")
class MotifPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.cfg = config

        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.OIV6_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.OIV6_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

        self.cnt = 0

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        if not self.training and self.cfg.MODEL.TRAIN_INFER:
            cat_labels = cat(rel_labels, dim=0)
            self.one_epoch(prod_rep, cat_labels)

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}
        return obj_dists, rel_dists, add_losses

    def one_epoch(self, rel_rep_norm, label):  # ONLY restore foreground samples when TRAIN_INFER is True
        self.cnt += 1
        fg_mask = label != 0
        flip_path = "infer_train_feat"
        path = os.path.join(self.cfg.OUTPUT_DIR, flip_path)
        os.makedirs(path, exist_ok=True)
        data = {
            'cnt': self.cnt,
            'rel_rep_norm': rel_rep_norm[fg_mask],
            'fg_label': label[fg_mask]
        }
        torch.save(data, os.path.join(path, "{}.pkl".format(self.cnt)))


@registry.ROI_RELATION_PREDICTOR.register("MotifPredictorGCM")
class MotifPredictorGCM(nn.Module):
    def __init__(self, config, in_channels, only_vis=False, only_txt=False):
        super(MotifPredictorGCM, self).__init__()
        self.cfg = config

        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.OIV6_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.OIV6_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)

        # modality erasing
        self.only_vis = only_vis
        self.only_txt = only_txt

        self.context_layer = LSTMContextPE(config, obj_classes, rel_classes, in_channels, self.only_vis, self.only_txt)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        # layer_init(self.rel_compress, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

        self.aug_num_rel_cls = int((len(rel_classes) - 1) * 1.1)         # augmented class number, α=0.1
        self.num_bg_cls = self.aug_num_rel_cls - (self.num_rel_cls - 1)  # bg class number
        self.uni_dist = (torch.ones(self.num_bg_cls) / self.num_bg_cls).cuda()

        self.embed_dim = 300
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)  # load Glove for predicates
        self.prefix_dim = 300

        # CoT generated semantic group, for VG
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            action_list = ['wearing', 'holding', 'sitting on', 'wears', 'riding', 'standing on', 'carrying', 'walking on',
                           'eating', 'using', 'laying on', 'playing', 'flying in']
            position_list = ['on', 'in', 'near', 'with', 'behind', 'above', 'under', 'in front of', 'at', 'attached to',
                             'over', 'between', 'along', 'across', 'against', 'on back of', 'lying on', 'walking in']
            possession_list = ['has', 'belonging to', 'for', 'part of', 'from']
            state_list = ['covering', 'covered in', 'mounted on', 'painted on', 'made of', 'growing on']
            descr_list = ['looking at', 'watching']
            conne_list = ['of', 'to', 'and']
            unclear_list = ['hanging from', 'parked on', 'says']

            # avg embedding within the same semantic group as the semantic group knowledge
            prefix_vecs = torch.randn([8, self.prefix_dim])
            prefix_vecs[1] = obj_edge_vectors(action_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[2] = obj_edge_vectors(position_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[3] = obj_edge_vectors(possession_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[4] = obj_edge_vectors(state_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[5] = obj_edge_vectors(descr_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[6] = obj_edge_vectors(conne_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[7] = obj_edge_vectors(unclear_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)

            action_ids = [rel_classes.index(i) for i in action_list]
            position_ids = [rel_classes.index(i) for i in position_list]
            possession_ids = [rel_classes.index(i) for i in possession_list]
            state_ids = [rel_classes.index(i) for i in state_list]
            descr_ids = [rel_classes.index(i) for i in descr_list]
            conne_ids = [rel_classes.index(i) for i in conne_list]
            unclear_ids = [rel_classes.index(i) for i in unclear_list]

            # predicates within the same semantic group have same knowledge
            prefix_embed = torch.Tensor(self.num_rel_cls, self.prefix_dim)
            prefix_embed.normal_(0, 1)
            prefix_embed[action_ids] = prefix_vecs[1]
            prefix_embed[position_ids] = prefix_vecs[2]
            prefix_embed[possession_ids] = prefix_vecs[3]
            prefix_embed[state_ids] = prefix_vecs[4]
            prefix_embed[descr_ids] = prefix_vecs[5]
            prefix_embed[conne_ids] = prefix_vecs[6]
            prefix_embed[unclear_ids] = prefix_vecs[7]

        # CoT generated semantic group, for GQA
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            action_list = ['wearing', 'holding', 'sitting on', 'riding', 'carrying', 'walking on', 'lying on', 'eating',
                           'watching', 'hanging on', 'using', 'playing with', 'throwing', 'catching', 'grazing on', 'cutting', 'feeding',
                           'running on', 'talking to', 'pulling', 'reaching for', 'skiing on', 'hitting', 'sitting at', 'leaning on',
                           'standing on', 'touching', 'swinging', 'talking on', 'pulled by', 'hang on']
            position_list = ['on', 'in', 'under', 'behind', 'in front of', 'next to', 'above', 'below', 'by', 'with',
                             'at', 'around', 'on the front of', 'on the side of', 'on the bottom of', 'beneath',
                             'crossing', 'standing next to', 'standing near', 'standing behind',
                             'surrounded by', 'standing by', 'standing in front of', 'close to', 'sitting next to',
                             'near', 'beside', 'on the back of', 'reflected in', 'facing',
                             'on top of', 'standing in', 'sitting in', 'lying on', 'parked along']
            state_list = ['covered by', 'covered in', 'filled with', 'mounted on', 'resting on', 'printed on',
                          'floating in', 'attached to', 'contain', 'surrounding', 'full of', 'hanging from',
                          'leaning against', 'covering', 'covered with', 'growing on', 'growing in', 'worn on']
            movement_list = ['flying in', 'flying', 'walking in', 'walking down', 'walking with', 'driving',
                             'driving on', 'driving down', 'swimming in', 'grazing in', 'parked on']
            inter_list = ['looking at', 'playing on', 'playing in', 'waiting for']
            unclear_list = ['of']

            # avg embedding within the same semantic group as the semantic group knowledge
            prefix_vecs = torch.randn([7, self.prefix_dim])
            prefix_vecs[1] = obj_edge_vectors(action_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[2] = obj_edge_vectors(position_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[3] = obj_edge_vectors(state_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[4] = obj_edge_vectors(movement_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[5] = obj_edge_vectors(inter_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[6] = obj_edge_vectors(unclear_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)

            action_ids = [rel_classes.index(i) for i in action_list]
            position_ids = [rel_classes.index(i) for i in position_list]
            state_ids = [rel_classes.index(i) for i in state_list]
            movement_ids = [rel_classes.index(i) for i in movement_list]
            inter_ids = [rel_classes.index(i) for i in inter_list]
            unclear_ids = [rel_classes.index(i) for i in unclear_list]

            # predicates within the same semantic group have same knowledge
            prefix_embed = torch.Tensor(self.num_rel_cls, self.prefix_dim)
            prefix_embed.normal_(0, 1)
            prefix_embed[action_ids] = prefix_vecs[1]
            prefix_embed[position_ids] = prefix_vecs[2]
            prefix_embed[state_ids] = prefix_vecs[3]
            prefix_embed[movement_ids] = prefix_vecs[4]
            prefix_embed[inter_ids] = prefix_vecs[5]
            prefix_embed[unclear_ids] = prefix_vecs[6]

        # CoT generated semantic group, for OIV6
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
            action_list = ['holds', 'wears', 'surf', 'hang', 'drink', 'holding_hands', 'ride', 'dance', 'skateboard',
                           'catch', 'highfive', 'eat', 'cut', 'handshake', 'kiss', 'talk_on_phone', 'throw', 'hits',
                           'kick', 'ski', 'plays', 'read', 'snowboard']
            position_list = ['at', 'on', 'under', 'inside_of']
            inter_list = ['interacts_with', 'hug']
            contain_list = ['contain']

            # avg embedding within the same semantic group as the semantic group knowledge
            prefix_vecs = torch.randn([5, self.prefix_dim])
            prefix_vecs[1] = obj_edge_vectors(action_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[2] = obj_edge_vectors(position_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[3] = obj_edge_vectors(inter_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[4] = obj_edge_vectors(contain_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)

            action_ids = [rel_classes.index(i) for i in action_list]
            position_ids = [rel_classes.index(i) for i in position_list]
            inter_ids = [rel_classes.index(i) for i in inter_list]
            contain_ids = [rel_classes.index(i) for i in contain_list]

            # predicates within the same semantic group have same knowledge
            prefix_embed = torch.Tensor(self.num_rel_cls, self.prefix_dim)
            prefix_embed.normal_(0, 1)
            prefix_embed[action_ids] = prefix_vecs[1]
            prefix_embed[position_ids] = prefix_vecs[2]
            prefix_embed[inter_ids] = prefix_vecs[3]
            prefix_embed[contain_ids] = prefix_vecs[4]

        self.rel_embed = nn.Embedding(self.aug_num_rel_cls, self.embed_dim)
        bg_rel_embed_vecs = torch.Tensor(self.num_bg_cls - 1, self.embed_dim)            # bg prototypes
        bg_rel_embed_vecs.normal_(0, 1)
        rel_embed_vecs = torch.cat([bg_rel_embed_vecs, rel_embed_vecs], dim=0)   # augmented prototypes
        self.rel_embed = nn.Embedding(self.aug_num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)

        self.prefix_vecs = prefix_vecs
        prefix_bg = torch.Tensor(self.num_bg_cls - 1, self.prefix_dim)
        prefix_bg.normal_(0, 1)
        prefix_embed = torch.cat([prefix_bg, prefix_embed], dim=0)
        self.prefix_embed = torch.nn.Parameter(prefix_embed)

        self.W_pred = MLP(self.embed_dim, self.pooling_dim, self.pooling_dim, 2)      # fusion of semantic group rep and rel rep
        self.project_head = MLP(self.pooling_dim, self.pooling_dim, self.pooling_dim, 2)
        dropout_p = 0.2
        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)

        file_path = '/all_mean_var_'
        if self.cfg.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            file_path += 'vg_'
        elif self.cfg.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            file_path += 'gqa_'
        elif self.cfg.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
            file_path += 'oiv6_'

        file_path += 'motifs.pkl'
        class_var = torch.load(self.cfg.OUTPUT_DIR + file_path, map_location=torch.device('cpu'))['all_var']
        class_var = F.normalize(class_var, p=2, dim=0)
        logit_tao = torch.cat([torch.ones([self.num_bg_cls]) * 0.07, class_var.sum(dim=1) / 10000], dim=0)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / logit_tao))

        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_sem_grp = 8
            self.sem_labels = torch.zeros(len(rel_classes)).cuda()
            self.sem_labels[action_ids] = 1
            self.sem_labels[position_ids] = 2
            self.sem_labels[possession_ids] = 3
            self.sem_labels[state_ids] = 4
            self.sem_labels[descr_ids] = 5
            self.sem_labels[conne_ids] = 6
            self.sem_labels[unclear_ids] = 7
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_sem_grp = 7
            self.sem_labels = torch.zeros(len(rel_classes)).cuda()
            self.sem_labels[action_ids] = 1
            self.sem_labels[position_ids] = 2
            self.sem_labels[state_ids] = 3
            self.sem_labels[movement_ids] = 4
            self.sem_labels[inter_ids] = 5
            self.sem_labels[unclear_ids] = 6
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
            self.num_sem_grp = 5
            self.sem_labels = torch.zeros(len(rel_classes)).cuda()
            self.sem_labels[action_ids] = 1
            self.sem_labels[position_ids] = 2
            self.sem_labels[inter_ids] = 3
            self.sem_labels[contain_ids] = 4

        # initialization of semantic group classifier
        self.sup_linear = nn.utils.weight_norm(nn.Linear(self.pooling_dim, self.num_sem_grp, bias=False))
        self.sup_linear.weight_g.data.fill_(1)
        self.sup_linear.weight_g.requires_grad = False

        self.sinkhorn = SinkhornKnopp()         # initialization of IGLR
        self.sup_compress = nn.Linear(self.prefix_dim + self.pooling_dim, self.pooling_dim)
        self.sup_loss = nn.CrossEntropyLoss()   # for semantic group classification
        self.norm_sup = nn.LayerNorm(self.prefix_dim + self.pooling_dim)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        add_losses = {}
        # refine object labels & encode context infomation
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                if not self.only_txt:
                    prod_rep = prod_rep * union_features

        # inject semantic group knowledge into rel rep
        predicate_proto = self.W_pred(self.prefix_embed * 0.2 + self.rel_embed.weight)

        prod_rep = self.project_head(self.dropout_rel(torch.relu(prod_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))

        # semantic group classification
        sup_dist = self.sup_linear(F.normalize(prod_rep, dim=1))
        sup_dist_logits = self.sinkhorn(sup_dist)   # IGLR for regularization

        if self.training:
            sem_labels = self.sem_labels[cat(rel_labels)]
            sup_cls_loss = self.sup_loss(sup_dist_logits, sem_labels.long())
            add_losses.update({'sup_cls_loss': sup_cls_loss * 0.1})

        sup_rep = sup_dist_logits @ self.prefix_vecs.cuda()
        prod_rep = self.sup_compress(self.norm_sup(torch.cat([sup_rep, prod_rep], dim=1)))
        prod_rep_norm = prod_rep / prod_rep.norm(dim=1, keepdim=True)  # r_norm
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm  (cosine similarity) & (Relation Prediction)
        rel_dists = prod_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  # <r_norm, c_norm> / τ

        if self.training and not self.only_vis and not self.only_txt:
            rel_labels = cat(rel_labels, dim=0)
            # separate fg and bg samples
            fg_id = rel_labels.nonzero().view(-1)
            fg_rel_labels = rel_labels[fg_id] + self.num_bg_cls - 1
            bg_id = (rel_labels == 0).nonzero().view(-1)
            num_bg = len(bg_id)
            bg_rel_labels = torch.randint(1, self.num_bg_cls, (num_bg,)).cuda()
            aug_rel_labels = torch.zeros_like(rel_labels)
            aug_rel_labels[fg_id] = fg_rel_labels
            aug_rel_labels[bg_id] = bg_rel_labels
            # uniform constraint for bg dists
            bg_rel_dists = rel_dists[bg_id][:, :self.num_bg_cls]  # reserve bg dists
            bg_rel_dists = (bg_rel_dists - bg_rel_dists.mean(dim=1).reshape(-1, 1)) / bg_rel_dists.std(dim=1).reshape(-1, 1)
            kl_loss = 10 * F.kl_div(bg_rel_dists.softmax(dim=-1).mean(dim=0).log(), self.uni_dist, reduction='mean')
            add_losses.update({"kl_loss": kl_loss})

        rel_dists = rel_dists[:, self.num_bg_cls - 1:]   # slice operation to return to the original class number

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor")
class VCTreePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.OIV6_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.OIV6_NUM_CLASSES

        self.cfg = config

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # learned-mixin
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        layer_init(self.ctx_compress, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)
        self.cnt = 0

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs,
                                                                          logger)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress(prod_rep * union_features)

        if not self.training and self.cfg.MODEL.TRAIN_INFER:
            cat_labels = cat(rel_labels, dim=0)
            self.one_epoch(prod_rep * union_features, cat_labels)

        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = ctx_dists + frq_dists

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

        return obj_dists, rel_dists, add_losses

    def one_epoch(self, rel_rep_norm, label):  # ONLY restore foreground samples when TRAIN_INFER is True
        self.cnt += 1
        fg_mask = label != 0
        flip_path = "infer_train_feat"
        path = os.path.join(self.cfg.OUTPUT_DIR, flip_path)
        os.makedirs(path, exist_ok=True)
        data = {
            'cnt': self.cnt,
            'rel_rep_norm': rel_rep_norm[fg_mask],
            'fg_label': label[fg_mask]
        }
        torch.save(data, os.path.join(path, "{}.pkl".format(self.cnt)))


@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictorGCM")
class VCTreePredictorGCM(nn.Module):
    def __init__(self, config, in_channels, only_vis=False, only_txt=False):
        super(VCTreePredictorGCM, self).__init__()
        self.cfg = config

        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.GQA_200_NUM_CLASSES
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
            self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.OIV6_NUM_CLASSES
            self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.OIV6_NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)

        # modality erasing
        self.only_vis = only_vis
        self.only_txt = only_txt

        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContextPE(config, obj_classes, rel_classes, statistics, in_channels,
                                                 self.only_vis, self.only_txt)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)

        self.aug_num_rel_cls = int((len(rel_classes) - 1) * 1.1)         # augmented class number, α=0.1 增广110%
        self.num_bg_cls = self.aug_num_rel_cls - (self.num_rel_cls - 1)  # bg class number
        self.uni_dist = (torch.ones(self.num_bg_cls) / self.num_bg_cls).cuda()

        self.embed_dim = 300
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR,
                                     wv_dim=self.embed_dim)  # load Glove for predicates
        self.prefix_dim = 300

        # CoT generated semantic group, for VG
        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            action_list = ['wearing', 'holding', 'sitting on', 'wears', 'riding', 'standing on', 'carrying',
                           'walking on', 'eating', 'using', 'laying on', 'playing', 'flying in']
            position_list = ['on', 'in', 'near', 'with', 'behind', 'above', 'under', 'in front of', 'at', 'attached to',
                             'over', 'between', 'along', 'across', 'against', 'on back of', 'lying on', 'walking in']
            possession_list = ['has', 'belonging to', 'for', 'part of', 'from']
            state_list = ['covering', 'covered in', 'mounted on', 'painted on', 'made of', 'growing on']
            descr_list = ['looking at', 'watching']
            conne_list = ['of', 'to', 'and']
            unclear_list = ['hanging from', 'parked on', 'says']

            # avg embedding within the same semantic group as the semantic group knowledge
            prefix_vecs = torch.randn([8, self.prefix_dim])
            prefix_vecs[1] = obj_edge_vectors(action_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[2] = obj_edge_vectors(position_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[3] = obj_edge_vectors(possession_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[4] = obj_edge_vectors(state_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[5] = obj_edge_vectors(descr_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[6] = obj_edge_vectors(conne_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[7] = obj_edge_vectors(unclear_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)

            action_ids = [rel_classes.index(i) for i in action_list]
            position_ids = [rel_classes.index(i) for i in position_list]
            possession_ids = [rel_classes.index(i) for i in possession_list]
            state_ids = [rel_classes.index(i) for i in state_list]
            descr_ids = [rel_classes.index(i) for i in descr_list]
            conne_ids = [rel_classes.index(i) for i in conne_list]
            unclear_ids = [rel_classes.index(i) for i in unclear_list]

            # predicates within the same semantic group have same knowledge
            prefix_embed = torch.Tensor(self.num_rel_cls, self.prefix_dim)
            prefix_embed.normal_(0, 1)
            prefix_embed[action_ids] = prefix_vecs[1]
            prefix_embed[position_ids] = prefix_vecs[2]
            prefix_embed[possession_ids] = prefix_vecs[3]
            prefix_embed[state_ids] = prefix_vecs[4]
            prefix_embed[descr_ids] = prefix_vecs[5]
            prefix_embed[conne_ids] = prefix_vecs[6]
            prefix_embed[unclear_ids] = prefix_vecs[7]

        # CoT generated semantic group, for GQA
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            action_list = ['wearing', 'holding', 'sitting on', 'riding', 'carrying', 'walking on', 'lying on', 'eating',
                           'watching', 'hanging on', 'using', 'playing with', 'throwing', 'catching', 'grazing on', 'cutting', 'feeding',
                           'running on', 'talking to', 'pulling', 'reaching for', 'skiing on', 'hitting', 'sitting at', 'leaning on', 'standing on',
                           'touching', 'swinging', 'talking on', 'pulled by', 'hang on']
            position_list = ['on', 'in', 'under', 'behind', 'in front of', 'next to', 'above', 'below', 'by', 'with',
                             'at', 'around', 'on the front of', 'on the side of', 'on the bottom of', 'beneath',
                             'crossing', 'standing next to', 'standing near', 'standing behind',
                             'surrounded by', 'standing by', 'standing in front of', 'close to', 'sitting next to',
                             'near', 'beside', 'on the back of', 'reflected in', 'facing',
                             'on top of', 'standing in', 'sitting in', 'lying on', 'parked along']
            state_list = ['covered by', 'covered in', 'filled with', 'mounted on', 'resting on', 'printed on',
                          'floating in', 'attached to', 'contain', 'surrounding', 'full of',
                          'hanging from', 'leaning against', 'covering', 'covered with', 'growing on',
                          'growing in', 'worn on']
            movement_list = ['flying in', 'flying', 'walking in', 'walking down', 'walking with', 'driving',
                             'driving on', 'driving down', 'swimming in', 'grazing in', 'parked on']
            inter_list = ['looking at', 'playing on', 'playing in', 'waiting for']
            unclear_list = ['of']

            # avg embedding within the same semantic group as the semantic group knowledge
            prefix_vecs = torch.randn([7, self.prefix_dim])
            prefix_vecs[1] = obj_edge_vectors(action_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[2] = obj_edge_vectors(position_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[3] = obj_edge_vectors(state_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[4] = obj_edge_vectors(movement_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[5] = obj_edge_vectors(inter_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[6] = obj_edge_vectors(unclear_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)

            action_ids = [rel_classes.index(i) for i in action_list]
            position_ids = [rel_classes.index(i) for i in position_list]
            state_ids = [rel_classes.index(i) for i in state_list]
            movement_ids = [rel_classes.index(i) for i in movement_list]
            inter_ids = [rel_classes.index(i) for i in inter_list]
            unclear_ids = [rel_classes.index(i) for i in unclear_list]

            # predicates within the same semantic group have same knowledge
            prefix_embed = torch.Tensor(self.num_rel_cls, self.prefix_dim)  # 51,300, 类型相同的前缀部分也相同
            prefix_embed.normal_(0, 1)
            prefix_embed[action_ids] = prefix_vecs[1]
            prefix_embed[position_ids] = prefix_vecs[2]
            prefix_embed[state_ids] = prefix_vecs[3]
            prefix_embed[movement_ids] = prefix_vecs[4]
            prefix_embed[inter_ids] = prefix_vecs[5]
            prefix_embed[unclear_ids] = prefix_vecs[6]

        # CoT generated semantic group, for OIV6
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
            action_list = ['holds', 'wears', 'surf', 'hang', 'drink', 'holding_hands', 'ride', 'dance', 'skateboard',
                           'catch', 'highfive', 'eat', 'cut', 'handshake', 'kiss', 'talk_on_phone', 'throw', 'hits',
                           'kick', 'ski', 'plays', 'read', 'snowboard']
            position_list = ['at', 'on', 'under', 'inside_of']
            inter_list = ['interacts_with', 'hug']
            contain_list = ['contain']

            # avg embedding within the same semantic group as the semantic group knowledge
            prefix_vecs = torch.randn([5, self.prefix_dim])
            prefix_vecs[1] = obj_edge_vectors(action_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[2] = obj_edge_vectors(position_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[3] = obj_edge_vectors(inter_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)
            prefix_vecs[4] = obj_edge_vectors(contain_list, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.prefix_dim).mean(dim=0)

            action_ids = [rel_classes.index(i) for i in action_list]
            position_ids = [rel_classes.index(i) for i in position_list]
            inter_ids = [rel_classes.index(i) for i in inter_list]
            contain_ids = [rel_classes.index(i) for i in contain_list]

            # predicates within the same semantic group have same knowledge
            prefix_embed = torch.Tensor(self.num_rel_cls, self.prefix_dim)
            prefix_embed.normal_(0, 1)
            prefix_embed[action_ids] = prefix_vecs[1]
            prefix_embed[position_ids] = prefix_vecs[2]
            prefix_embed[inter_ids] = prefix_vecs[3]
            prefix_embed[contain_ids] = prefix_vecs[4]

        self.rel_embed = nn.Embedding(self.aug_num_rel_cls, self.embed_dim)
        bg_rel_embed_vecs = torch.Tensor(self.num_bg_cls - 1, self.embed_dim)            # bg prototypes
        bg_rel_embed_vecs.normal_(0, 1)
        rel_embed_vecs = torch.cat([bg_rel_embed_vecs, rel_embed_vecs], dim=0)   # augmented prototypes
        self.rel_embed = nn.Embedding(self.aug_num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)

        self.prefix_vecs = prefix_vecs
        prefix_bg = torch.Tensor(self.num_bg_cls - 1, self.prefix_dim)
        prefix_bg.normal_(0, 1)
        prefix_embed = torch.cat([prefix_bg, prefix_embed], dim=0)
        self.prefix_embed = torch.nn.Parameter(prefix_embed)

        self.W_pred = MLP(self.embed_dim, self.pooling_dim, self.pooling_dim, 2)      # fusion of semantic group rep and rel rep
        self.project_head = MLP(self.pooling_dim, self.pooling_dim, self.pooling_dim, 2)
        dropout_p = 0.2
        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)

        file_path = '/all_mean_var_'
        if self.cfg.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            file_path += 'vg_'
        elif self.cfg.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            file_path += 'gqa_'
        elif self.cfg.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
            file_path += 'oiv6_'

        file_path += 'vctree.pkl'
        class_var = torch.load(self.cfg.OUTPUT_DIR + file_path, map_location=torch.device('cpu'))['all_var']
        class_var = F.normalize(class_var, p=2, dim=0)
        logit_tao = torch.cat([torch.ones([self.num_bg_cls]) * 0.07, class_var.sum(dim=1) / 10000], dim=0)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / logit_tao))

        if config.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
            self.num_sem_grp = 8
            self.sem_labels = torch.zeros(len(rel_classes)).cuda()
            self.sem_labels[action_ids] = 1
            self.sem_labels[position_ids] = 2
            self.sem_labels[possession_ids] = 3
            self.sem_labels[state_ids] = 4
            self.sem_labels[descr_ids] = 5
            self.sem_labels[conne_ids] = 6
            self.sem_labels[unclear_ids] = 7
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
            self.num_sem_grp = 7
            self.sem_labels = torch.zeros(len(rel_classes)).cuda()
            self.sem_labels[action_ids] = 1
            self.sem_labels[position_ids] = 2
            self.sem_labels[state_ids] = 3
            self.sem_labels[movement_ids] = 4
            self.sem_labels[inter_ids] = 5
            self.sem_labels[unclear_ids] = 6
        elif config.GLOBAL_SETTING.DATASET_CHOICE == 'OIV6':
            self.num_sem_grp = 5
            self.sem_labels = torch.zeros(len(rel_classes)).cuda()
            self.sem_labels[action_ids] = 1
            self.sem_labels[position_ids] = 2
            self.sem_labels[inter_ids] = 3
            self.sem_labels[contain_ids] = 4

        # initialization of semantic group classifier
        self.sup_linear = nn.utils.weight_norm(nn.Linear(self.pooling_dim, self.num_sem_grp, bias=False))
        self.sup_linear.weight_g.data.fill_(1)
        self.sup_linear.weight_g.requires_grad = False

        self.sinkhorn = SinkhornKnopp()           # initialization of IGLR
        self.sup_compress = nn.Linear(self.prefix_dim + self.pooling_dim, self.pooling_dim)
        self.sup_loss = nn.CrossEntropyLoss()    # for semantic group classification
        self.norm_sup = nn.LayerNorm(self.prefix_dim + self.pooling_dim)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        add_losses = {}
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if not self.only_txt:
            prod_rep = prod_rep * union_features

        # inject semantic group knowledge into rel rep
        predicate_proto = self.W_pred(self.prefix_embed * 0.2 + self.rel_embed.weight)
        prod_rep = self.project_head(self.dropout_rel(torch.relu(prod_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))

        # semantic group classification
        sup_dist = self.sup_linear(F.normalize(prod_rep, dim=1))
        sup_dist_logits = self.sinkhorn(sup_dist)   # IGLR for regularization

        if self.training:
            sem_labels = self.sem_labels[cat(rel_labels)]
            sup_cls_loss = self.sup_loss(sup_dist_logits, sem_labels.long())
            add_losses.update({'sup_cls_loss': sup_cls_loss * 0.1})

        sup_rep = sup_dist_logits @ self.prefix_vecs.cuda()
        prod_rep = self.sup_compress(self.norm_sup(torch.cat([sup_rep, prod_rep], dim=1)))
        prod_rep_norm = prod_rep / prod_rep.norm(dim=1, keepdim=True)  # r_norm
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm  (cosine similarity) & (Relation Prediction)
        rel_dists = prod_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  # <r_norm, c_norm> / τ

        if self.training and not self.only_vis and not self.only_txt:
            rel_labels = cat(rel_labels, dim=0)
            # separate fg and bg samples
            fg_id = rel_labels.nonzero().view(-1)
            fg_rel_labels = rel_labels[fg_id] + self.num_bg_cls - 1
            bg_id = (rel_labels == 0).nonzero().view(-1)
            num_bg = len(bg_id)
            bg_rel_labels = torch.randint(1, self.num_bg_cls, (num_bg,)).cuda()
            aug_rel_labels = torch.zeros_like(rel_labels)
            aug_rel_labels[fg_id] = fg_rel_labels
            aug_rel_labels[bg_id] = bg_rel_labels
            # uniform constraint for bg dists
            bg_rel_dists = rel_dists[bg_id][:, :self.num_bg_cls]  # reserve bg dists
            bg_rel_dists = (bg_rel_dists - bg_rel_dists.mean(dim=1).reshape(-1, 1)) / bg_rel_dists.std(dim=1).reshape(-1, 1)
            kl_loss = 10 * F.kl_div(bg_rel_dists.softmax(dim=-1).mean(dim=0).log(), self.uni_dist, reduction='mean')
            add_losses.update({"kl_loss": kl_loss})

        rel_dists = rel_dists[:, self.num_bg_cls - 1:]   # slice operation to return to the original class number

        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = rel_dists + frq_dists

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("CausalAnalysisPredictor")
class CausalAnalysisPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(CausalAnalysisPredictor, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.fusion_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE
        self.separate_spatial = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        self.use_vtranse = config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse"
        self.effect_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        if config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "motifs":
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vctree":
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse":
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            print('ERROR: Invalid Context Layer')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        if self.use_vtranse:
            self.edge_dim = self.pooling_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.pooling_dim * 2)
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=False)
        else:
            self.edge_dim = self.hidden_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            self.post_cat = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                                            nn.ReLU(inplace=True), ])
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.vis_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)

        if self.fusion_type == 'gate':
            self.ctx_gate_fc = nn.Linear(self.pooling_dim, self.num_rel_cls)
            layer_init(self.ctx_gate_fc, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        if not self.use_vtranse:
            layer_init(self.post_cat[0], xavier=True)
            layer_init(self.ctx_compress, xavier=True)
        layer_init(self.vis_compress, xavier=True)

        assert self.pooling_dim == config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        # convey statistics into FrequencyBias to avoid loading again
        self.freq_bias = FrequencyBias(config, statistics)

        # add spatial emb for visual feature
        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(*[nn.Linear(32, self.hidden_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(self.hidden_dim, self.pooling_dim),
                                           nn.ReLU(inplace=True)
                                           ])
            layer_init(self.spt_emb[0], xavier=True)
            layer_init(self.spt_emb[2], xavier=True)

        self.label_smooth_loss = Label_Smoothing_Regression(e=1.0)

        # untreated average features
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        self.average_ratio = 0.0005

        self.register_buffer("untreated_spt", torch.zeros(32))
        self.register_buffer("untreated_conv_spt", torch.zeros(self.pooling_dim))
        self.register_buffer("avg_post_ctx", torch.zeros(self.pooling_dim))
        self.register_buffer("untreated_feat", torch.zeros(self.pooling_dim))

    def pair_feature_generate(self, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger,
                              ctx_average=False):
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs,
                                                                          logger, ctx_average=ctx_average)
        obj_dist_prob = F.softmax(obj_dists, dim=-1)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dists.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(rel_pair_idxs, head_reps, tail_reps,
                                                                             obj_preds, obj_boxs, obj_prob_list):
            if self.use_vtranse:
                ctx_reps.append(head_rep[pair_idx[:, 0]] - tail_rep[pair_idx[:, 1]])
            else:
                ctx_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
            pair_obj_probs.append(torch.stack((obj_prob[pair_idx[:, 0]], obj_prob[pair_idx[:, 1]]), dim=2))
            pair_bboxs_info.append(get_box_pair_info(obj_box[pair_idx[:, 0]], obj_box[pair_idx[:, 1]]))
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        ctx_rep = cat(ctx_reps, dim=0)
        if self.use_vtranse:
            post_ctx_rep = ctx_rep
        else:
            post_ctx_rep = self.post_cat(ctx_rep)

        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list = self.pair_feature_generate(
            roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger)

        if (not self.training) and self.effect_analysis:
            with torch.no_grad():
                avg_post_ctx_rep, _, _, avg_pair_obj_prob, _, _, _, _ = self.pair_feature_generate(roi_features,
                                                                                                   proposals,
                                                                                                   rel_pair_idxs,
                                                                                                   num_objs, obj_boxs,
                                                                                                   logger,
                                                                                                   ctx_average=True)

        if self.separate_spatial:
            union_features, spatial_conv_feats = union_features
            post_ctx_rep = post_ctx_rep * spatial_conv_feats

        if self.spatial_for_vision:
            post_ctx_rep = post_ctx_rep * self.spt_emb(pair_bbox)

        rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_pred, use_label_dist=False)
        rel_dist_list = rel_dists.split(num_rels, dim=0)

        add_losses = {}
        # additional loss
        if self.training:
            rel_labels = cat(rel_labels, dim=0)

            # binary loss for VCTree
            if binary_preds is not None:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

            # branch constraint: make sure each branch can predict independently
            add_losses['auxiliary_ctx'] = F.cross_entropy(self.ctx_compress(post_ctx_rep), rel_labels)
            if not (self.fusion_type == 'gate'):
                add_losses['auxiliary_vis'] = F.cross_entropy(self.vis_compress(union_features), rel_labels)
                add_losses['auxiliary_frq'] = F.cross_entropy(self.freq_bias.index_with_labels(pair_pred.long()),
                                                              rel_labels)

            # untreated average feature
            if self.spatial_for_vision:
                self.untreated_spt = self.moving_average(self.untreated_spt, pair_bbox)
            if self.separate_spatial:
                self.untreated_conv_spt = self.moving_average(self.untreated_conv_spt, spatial_conv_feats)
            self.avg_post_ctx = self.moving_average(self.avg_post_ctx, post_ctx_rep)
            self.untreated_feat = self.moving_average(self.untreated_feat, union_features)

        elif self.effect_analysis:
            with torch.no_grad():
                # untreated spatial
                if self.spatial_for_vision:
                    avg_spt_rep = self.spt_emb(self.untreated_spt.clone().detach().view(1, -1))
                # untreated context
                avg_ctx_rep = avg_post_ctx_rep * avg_spt_rep if self.spatial_for_vision else avg_post_ctx_rep
                avg_ctx_rep = avg_ctx_rep * self.untreated_conv_spt.clone().detach().view(1,
                                                                                          -1) if self.separate_spatial else avg_ctx_rep
                # untreated visual
                avg_vis_rep = self.untreated_feat.clone().detach().view(1, -1)
                # untreated category dist
                avg_frq_rep = avg_pair_obj_prob

            if self.effect_type == 'TDE':  # TDE of CTX
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(
                    union_features, avg_ctx_rep, pair_obj_probs)
            elif self.effect_type == 'NIE':  # NIE of FRQ
                rel_dists = self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs) - self.calculate_logits(
                    union_features, avg_ctx_rep, avg_frq_rep)
            elif self.effect_type == 'TE':  # Total Effect
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(
                    union_features, avg_ctx_rep, avg_frq_rep)
            else:
                assert self.effect_type == 'none'
                pass
            rel_dist_list = rel_dists.split(num_rels, dim=0)

        return obj_dist_list, rel_dist_list, add_losses

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def calculate_logits(self, vis_rep, ctx_rep, frq_rep, use_label_dist=True, mean_ctx=False):
        if use_label_dist:
            frq_dists = self.freq_bias.index_with_probability(frq_rep)
        else:
            frq_dists = self.freq_bias.index_with_labels(frq_rep.long())

        if mean_ctx:
            ctx_rep = ctx_rep.mean(-1).unsqueeze(-1)
        vis_dists = self.vis_compress(vis_rep)
        ctx_dists = self.ctx_compress(ctx_rep)

        if self.fusion_type == 'gate':
            ctx_gate_dists = self.ctx_gate_fc(ctx_rep)
            union_dists = ctx_dists * torch.sigmoid(vis_dists + frq_dists + ctx_gate_dists)
            # union_dists = (ctx_dists.exp() * torch.sigmoid(vis_dists + frq_dists + ctx_constraint) + 1e-9).log()    # improve on zero-shot, but low mean recall and TDE recall
            # union_dists = ctx_dists * torch.sigmoid(vis_dists * frq_dists)                                          # best conventional Recall results
            # union_dists = (ctx_dists.exp() + vis_dists.exp() + frq_dists.exp() + 1e-9).log()                        # good zero-shot Recall
            # union_dists = ctx_dists * torch.max(torch.sigmoid(vis_dists), torch.sigmoid(frq_dists))                 # good zero-shot Recall
            # union_dists = ctx_dists * torch.sigmoid(vis_dists) * torch.sigmoid(frq_dists)                           # balanced recall and mean recall
            # union_dists = ctx_dists * (torch.sigmoid(vis_dists) + torch.sigmoid(frq_dists)) / 2.0                   # good zero-shot Recall
            # union_dists = ctx_dists * torch.sigmoid((vis_dists.exp() + frq_dists.exp() + 1e-9).log())               # good zero-shot Recall, bad for all of the rest

        elif self.fusion_type == 'sum':
            union_dists = vis_dists + ctx_dists + frq_dists
        else:
            print('invalid fusion type')

        return union_dists

    def binary_ce_loss(self, logits, gt):
        batch_size, num_cat = logits.shape
        answer = torch.zeros((batch_size, num_cat), device=gt.device).float()
        answer[torch.arange(batch_size, device=gt.device), gt.long()] = 1.0
        return F.binary_cross_entropy_with_logits(logits, answer) * num_cat

    def fusion(self, x, y):
        return F.relu(x + y) - (x - y) ** 2


def make_roi_relation_predictor(cfg, in_channels, only_vis=False, only_txt=False):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels, only_vis, only_txt)
